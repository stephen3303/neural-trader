import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from src.model.network import ModelConfig, TradingNet
from src.training.trainer import ContinualTrainer, TrainerConfig, find_latest_checkpoint

N_FEATURES = 15
WINDOW = 20


def _model():
    return TradingNet(ModelConfig(n_features=N_FEATURES, window=WINDOW, hidden_size=8, trunk_size=8))


def test_find_latest_checkpoint_returns_none_when_dir_missing(tmp_path):
    assert find_latest_checkpoint(tmp_path / "does-not-exist") is None


def test_find_latest_checkpoint_returns_none_when_dir_empty(tmp_path):
    d = tmp_path / "checkpoints"
    d.mkdir()
    assert find_latest_checkpoint(d) is None


def test_find_latest_checkpoint_picks_highest_version_not_newest_mtime(tmp_path):
    d = tmp_path / "checkpoints"
    d.mkdir()
    # Write v10 before v2 so mtime order is the OPPOSITE of version order --
    # version (from the filename) must win, not mtime.
    (d / "model_v10.pt").write_bytes(b"x")
    (d / "model_v2.pt").write_bytes(b"x")
    (d / "model_v9.pt").write_bytes(b"x")
    assert find_latest_checkpoint(d) == d / "model_v10.pt"


def test_find_latest_checkpoint_ignores_non_matching_files(tmp_path):
    d = tmp_path / "checkpoints"
    d.mkdir()
    (d / "model_v1.pt").write_bytes(b"x")
    (d / "model_v1.json").write_bytes(b"{}")  # metadata, not a checkpoint -- different suffix
    (d / "notes.txt").write_bytes(b"hi")
    assert find_latest_checkpoint(d) == d / "model_v1.pt"


def test_trainer_round_trips_promoted_weights_through_a_checkpoint(tmp_path):
    """The actual bug this closes: nothing previously loaded _save_checkpoint's
    output back on startup, so every process restart silently discarded all
    promoted retraining and resumed from an untrained v0 model. This proves
    a promoted model's weights (not just its version number) survive a
    save/load round trip into a FRESH trainer, simulating a process
    restart."""
    ckpt_dir = tmp_path / "checkpoints"
    model = _model()
    trainer = ContinualTrainer(model, TrainerConfig(checkpoint_dir=str(ckpt_dir)))

    # Simulate a promotion exactly like maybe_retrain() does: give the live
    # model distinguishable weights, then save a checkpoint for it.
    with torch.no_grad():
        model.action_head.bias[:] = torch.tensor([3.0, -3.0, -3.0])
    trainer._version = 5
    trainer._save_checkpoint({"version": 5, "promoted": True})

    # A brand-new trainer with a brand-new (untrained) model -- standing in
    # for a freshly started process.
    fresh_model = _model()
    with torch.no_grad():
        fresh_model.action_head.bias[:] = torch.tensor([0.0, 0.0, 0.0])
    fresh_trainer = ContinualTrainer(fresh_model, TrainerConfig(checkpoint_dir=str(ckpt_dir)))

    resumed_version = fresh_trainer.load_latest_checkpoint()

    assert resumed_version == 5
    assert fresh_trainer.model_version() == "v5"
    assert torch.allclose(fresh_trainer.model.action_head.bias, torch.tensor([3.0, -3.0, -3.0]))
    # Loading must mutate the model IN PLACE (load_state_dict), not rebind
    # self.model to a new object -- any other code already holding a
    # reference to fresh_trainer.model (e.g. Orchestrator) must see the
    # resumed weights too.
    assert fresh_trainer.model is fresh_model


def test_load_latest_checkpoint_returns_none_with_no_checkpoints(tmp_path):
    model = _model()
    trainer = ContinualTrainer(model, TrainerConfig(checkpoint_dir=str(tmp_path / "checkpoints")))
    assert trainer.load_latest_checkpoint() is None
    assert trainer.model_version() == "v0"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
