"""
Replay buffer for continual / online learning.

This is the piece that makes "continuously monitor the market to supervise
training" concrete: every time a prediction's forward-return horizon
elapses, the realized outcome becomes a *label* and gets appended here.
The trainer later samples from this buffer to fine-tune the network -- no
human labeling required, but every sample is still a real, dated market
outcome.

Two properties matter for safety/quality and are both handled here:

1. Recency bias with a long memory ("stability-plasticity" tradeoff).
   Pure "train on the last N samples" causes catastrophic forgetting of
   older regimes (e.g. the model forgets how high-vol days behave after a
   long calm stretch). Pure "train on everything ever seen" makes the
   model slow to adapt to genuine regime change. `sample_batch()` mixes a
   configurable fraction of recent data with a uniform sample of the full
   history to balance the two.

2. Bounded memory. The buffer is capped (`max_size`); once full it evicts
   oldest samples first (ring buffer), so this can run indefinitely without
   unbounded RAM/disk growth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ReplayBuffer:
    window: int
    n_features: int
    max_size: int = 200_000
    recent_fraction: float = 0.6   # fraction of each training batch drawn from the most recent slice
    recent_window_frac: float = 0.15  # what counts as "recent" = last X% of the buffer

    X: np.ndarray = field(init=False)
    y_action: np.ndarray = field(init=False)
    y_ret: np.ndarray = field(init=False)
    tickers: list = field(default_factory=list, init=False)
    timestamps: list = field(default_factory=list, init=False)
    _size: int = field(default=0, init=False)
    _write_ptr: int = field(default=0, init=False)

    def __post_init__(self):
        self.X = np.zeros((self.max_size, self.window, self.n_features), dtype=np.float32)
        self.y_action = np.zeros((self.max_size,), dtype=np.int64)
        self.y_ret = np.zeros((self.max_size,), dtype=np.float32)
        self.tickers = [None] * self.max_size
        self.timestamps = [None] * self.max_size

    def __len__(self) -> int:
        return self._size

    def add(self, x: np.ndarray, action: int, fwd_ret: float, ticker: str, timestamp) -> None:
        i = self._write_ptr
        self.X[i] = x
        self.y_action[i] = action
        self.y_ret[i] = fwd_ret
        self.tickers[i] = ticker
        self.timestamps[i] = timestamp
        self._write_ptr = (i + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

    def add_many(self, X: np.ndarray, y_action: np.ndarray, y_ret: np.ndarray,
                 ticker: str, timestamps) -> None:
        for x, a, r, ts in zip(X, y_action, y_ret, timestamps):
            self.add(x, int(a), float(r), ticker, ts)

    def sample_batch(self, batch_size: int, rng: np.random.Generator | None = None):
        if self._size == 0:
            raise ValueError("ReplayBuffer is empty")
        rng = rng or np.random.default_rng()
        n_recent = int(batch_size * self.recent_fraction)
        n_uniform = batch_size - n_recent

        recent_span = max(1, int(self._size * self.recent_window_frac))
        # valid (written) indices are [0, self._size) in insertion order when
        # not yet wrapped; once wrapped, "recent" = the slice just behind the
        # write pointer.
        if self._size < self.max_size:
            recent_idx = rng.integers(max(0, self._size - recent_span), self._size, size=n_recent)
            uniform_idx = rng.integers(0, self._size, size=n_uniform)
        else:
            start = (self._write_ptr - recent_span) % self.max_size
            recent_idx = (start + rng.integers(0, recent_span, size=n_recent)) % self.max_size
            uniform_idx = rng.integers(0, self.max_size, size=n_uniform)

        idx = np.concatenate([recent_idx, uniform_idx])
        return self.X[idx], self.y_action[idx], self.y_ret[idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(
            path, X=self.X[: self._size], y_action=self.y_action[: self._size],
            y_ret=self.y_ret[: self._size],
            tickers=np.array(self.tickers[: self._size], dtype=object),
            timestamps=np.array([str(t) for t in self.timestamps[: self._size]], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path, window: int, n_features: int, max_size: int = 200_000) -> "ReplayBuffer":
        data = np.load(path, allow_pickle=True)
        buf = cls(window=window, n_features=n_features, max_size=max_size)
        n = min(len(data["X"]), max_size)
        buf.add_many(data["X"][-n:], data["y_action"][-n:], data["y_ret"][-n:],
                     ticker=None, timestamps=data["timestamps"][-n:])
        # restore per-sample tickers (add_many above overwrote with a single value)
        for i in range(n):
            buf.tickers[i] = data["tickers"][-n:][i]
        return buf
