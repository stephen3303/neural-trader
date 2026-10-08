"""
Continual (online) training loop with a champion/challenger safety gate.

This is the core of "continuously monitor the market... to supervise its
training": as realized outcomes accumulate in the ReplayBuffer, this
trainer periodically fine-tunes a *copy* of the live model (the
"challenger"), validates it on the most recent held-out slice of data, and
only promotes it to be the new live model (the "champion") if it actually
performs at least as well out-of-sample.

Why the champion/challenger gate matters: naive continual learning -- just
keep calling `.backward()` on the live model forever -- has no safeguard
against a bad batch of data (a data glitch, a flash-crash bar, a buggy
feature) quietly making the live model worse. Training a challenger copy
and gating promotion on held-out validation means a bad update can at
worst leave the champion unchanged; it should never actively worsen it
through an unguarded in-place update. The validation split is
time-ordered (most recent N samples), not randomly shuffled, because
shuffled validation on financial time series leaks future information
into the "held out" set and makes the gate meaningless.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.buffer import ReplayBuffer
from src.model.network import TradingNet, compute_loss


@dataclass
class TrainerConfig:
    min_buffer_size: int = 500       # don't retrain until the buffer has at least this many samples
    retrain_every_n_new: int = 200   # retrain after this many *new* samples arrive
    epochs_per_retrain: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    val_fraction: float = 0.15       # most-recent slice of the buffer reserved for validation
    max_val_regression: float = 0.02  # allow the challenger to be slightly worse before vetoing promotion
    checkpoint_dir: str = "checkpoints"


class ContinualTrainer:
    def __init__(self, model: TradingNet, cfg: TrainerConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self._samples_since_retrain = 0
        self._version = 0
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

    def notify_new_samples(self, n: int) -> None:
        self._samples_since_retrain += n

    def ready_to_retrain(self, buffer: ReplayBuffer) -> bool:
        return (
            len(buffer) >= self.cfg.min_buffer_size
            and self._samples_since_retrain >= self.cfg.retrain_every_n_new
        )

    def _validation_slice(self, buffer: ReplayBuffer):
        n_val = max(1, int(len(buffer) * self.cfg.val_fraction))
        # most recently written n_val samples, respecting the ring-buffer wraparound
        if len(buffer) < buffer.max_size:
            start = max(0, len(buffer) - n_val)
            idx = np.arange(start, len(buffer))
        else:
            idx = (buffer._write_ptr - np.arange(1, n_val + 1)) % buffer.max_size
        return buffer.X[idx], buffer.y_action[idx], buffer.y_ret[idx]

    @staticmethod
    def _evaluate(model: nn.Module, X: np.ndarray, y_action: np.ndarray, y_ret: np.ndarray, device: str) -> dict:
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X).to(device)
            yat = torch.from_numpy(y_action).to(device)
            yrt = torch.from_numpy(y_ret).to(device)
            logits, pred_ret = model(xt)
            _, loss_parts = compute_loss(logits, pred_ret, yat, yrt)
            acc = (logits.argmax(dim=-1) == yat).float().mean().item()
        return {"ce": loss_parts["ce"], "mse": loss_parts["mse"], "accuracy": acc}

    def maybe_retrain(self, buffer: ReplayBuffer) -> dict | None:
        """Returns a dict describing what happened if a retrain was run
        (whether or not it was promoted), or None if it skipped retraining."""
        if not self.ready_to_retrain(buffer):
            return None

        val_X, val_yA, val_yR = self._validation_slice(buffer)
        champion_val = self._evaluate(self.model, val_X, val_yA, val_yR, self.device)

        challenger = copy.deepcopy(self.model)
        optimizer = torch.optim.Adam(challenger.parameters(), lr=self.cfg.lr)
        challenger.train()
        rng = np.random.default_rng()

        n_batches = max(1, len(buffer) // self.cfg.batch_size)
        for _epoch in range(self.cfg.epochs_per_retrain):
            for _ in range(n_batches):
                X, yA, yR = buffer.sample_batch(self.cfg.batch_size, rng)
                xt = torch.from_numpy(X).to(self.device)
                yat = torch.from_numpy(yA).to(self.device)
                yrt = torch.from_numpy(yR).to(self.device)
                optimizer.zero_grad()
                logits, pred_ret = challenger(xt)
                loss, _ = compute_loss(logits, pred_ret, yat, yrt)
                loss.backward()
                nn.utils.clip_grad_norm_(challenger.parameters(), max_norm=1.0)
                optimizer.step()

        challenger_val = self._evaluate(challenger, val_X, val_yA, val_yR, self.device)

        # Promote only if the challenger isn't meaningfully worse on the
        # held-out, time-ordered validation slice. "Worse" is judged on
        # combined loss (ce + mse), not accuracy alone, since accuracy
        # ignores the regression head and is noisy at small val sizes.
        champion_loss = champion_val["ce"] + champion_val["mse"]
        challenger_loss = challenger_val["ce"] + challenger_val["mse"]
        promoted = challenger_loss <= champion_loss * (1 + self.cfg.max_val_regression)

        if promoted:
            self.model = challenger
            self._version += 1

        record = {
            "timestamp": time.time(),
            "version": self._version,
            "promoted": promoted,
            "champion_val": champion_val,
            "challenger_val": challenger_val,
            "buffer_size": len(buffer),
        }
        self.history.append(record)
        self._samples_since_retrain = 0

        if promoted:
            self._save_checkpoint(record)

        return record

    def _save_checkpoint(self, record: dict) -> None:
        ckpt_path = Path(self.cfg.checkpoint_dir) / f"model_v{self._version}.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        meta_path = Path(self.cfg.checkpoint_dir) / f"model_v{self._version}.json"
        meta_path.write_text(json.dumps(record, indent=2))

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a saved champion's weights (from `_save_checkpoint`) into
        the live model IN PLACE -- `self.model.load_state_dict(...)`, never
        `self.model = ...` -- so this is safe to call before or after
        anything else holds a reference to `self.model` (see
        Orchestrator._try_predict's comment on why rebinding vs. mutating
        matters here).

        Returns the version number parsed from the filename (e.g. 3 for
        `model_v3.pt`) and also updates `self._version` to it, so
        `model_version()` reports accurately post-resume. Raises
        FileNotFoundError / ValueError if `path` doesn't look like a
        checkpoint this trainer wrote."""
        path = Path(path)
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self._version = _parse_checkpoint_version(path)
        # The in-memory buffer/sample counters are always empty right after
        # a process restart, so there's nothing to resume there -- only the
        # model weights persist across a restart.
        self._samples_since_retrain = 0
        return self._version

    def load_latest_checkpoint(self) -> int | None:
        """Convenience wrapper: find and load the highest-version checkpoint
        in `self.cfg.checkpoint_dir`, if any exist. Returns the loaded
        version number, or None if there's no checkpoint to resume from
        (e.g. first-ever run, or a fresh checkpoint_dir)."""
        latest = find_latest_checkpoint(self.cfg.checkpoint_dir)
        if latest is None:
            return None
        return self.load_checkpoint(latest)

    def model_version(self) -> str:
        return f"v{self._version}"


def _parse_checkpoint_version(path: Path) -> int:
    # Filenames are always "model_v{N}.pt", written only by _save_checkpoint.
    stem = path.stem  # "model_v3"
    try:
        return int(stem.rsplit("_v", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"'{path}' doesn't look like a model_v<N>.pt checkpoint") from exc


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the highest-version `model_v*.pt` file in `checkpoint_dir`,
    or None if the directory doesn't exist or has no checkpoints yet.
    Version is taken from the filename, not file mtime, since mtime can be
    disturbed by copying/syncing checkpoints between machines."""
    d = Path(checkpoint_dir)
    if not d.is_dir():
        return None
    candidates = []
    for p in d.glob("model_v*.pt"):
        try:
            candidates.append((_parse_checkpoint_version(p), p))
        except ValueError:
            continue  # ignore anything that doesn't match the naming convention
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]
