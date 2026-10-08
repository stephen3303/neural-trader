import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from src.data.buffer import ReplayBuffer
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


def _filled_buffer(tmp_path, y_action, X=None):
    """A ReplayBuffer pre-loaded with exactly the given action labels
    (feature window content doesn't matter for the class-weighting tests,
    and is deliberately all-zero for the degeneracy tests below)."""
    n = len(y_action)
    buf = ReplayBuffer(window=WINDOW, n_features=N_FEATURES, max_size=max(n, 10))
    if X is None:
        X = np.zeros((n, WINDOW, N_FEATURES), dtype=np.float32)
    y_ret = np.zeros(n, dtype=np.float32)
    ts = list(range(n))
    buf.add_many(X, np.asarray(y_action, dtype=np.int64), y_ret, ticker="T", timestamps=ts)
    return buf


class TestClassWeights:
    """Regression coverage for a real bug found by running the online loop
    for 3000 bars against this exact codebase (not a demo log): "hold" is
    under 3% of labels under the default deadband, and unweighted
    cross-entropy gave the model essentially no incentive to ever predict
    it -- hold was predicted in 1 of 14 model versions seen (the untrained
    v0) and zero times across 2600+ predictions after any real training."""

    def test_rare_class_gets_a_higher_weight_than_common_classes(self, tmp_path):
        from src.training.trainer import ContinualTrainer, TrainerConfig

        # 600 sell, 50 hold, 350 buy -- skewed like the default deadband's
        # real distribution, but not so extreme that both minority classes
        # saturate the max_class_weight clip (that's covered separately
        # below) and become indistinguishable from each other.
        y = [0] * 600 + [1] * 50 + [2] * 350
        buf = _filled_buffer(tmp_path, y)
        trainer = ContinualTrainer(_model(), TrainerConfig(checkpoint_dir=str(tmp_path / "ckpt")))

        weights = trainer._class_weights(buf)

        assert weights is not None
        w = weights.numpy()
        # hold (rarest) must outweigh both sell and buy; sell (most common)
        # must be the smallest weight.
        assert w[1] > w[2] > w[0]
        # Normalized to average 1.0 so the overall loss scale is stable
        # across retrains regardless of how skewed the buffer is.
        assert w.mean() == pytest.approx(1.0, abs=1e-5)

    def test_extreme_imbalance_is_clipped_not_unbounded(self, tmp_path):
        from src.training.trainer import ContinualTrainer, TrainerConfig

        y = [0] * 9999 + [1] * 1  # a single hold sample in ~10k otherwise
        buf = _filled_buffer(tmp_path, y)
        cfg = TrainerConfig(checkpoint_dir=str(tmp_path / "ckpt"), max_class_weight=10.0)
        trainer = ContinualTrainer(_model(), cfg)

        weights = trainer._class_weights(buf)

        # Pre-clip, hold's raw inverse-frequency weight here would be in
        # the thousands; it must never be allowed to dominate the loss
        # from a single noisy sample like that.
        assert weights.numpy().max() <= cfg.max_class_weight * 1.01

    def test_disabled_via_config_returns_none(self, tmp_path):
        from src.training.trainer import ContinualTrainer, TrainerConfig

        buf = _filled_buffer(tmp_path, [0, 1, 2] * 20)
        cfg = TrainerConfig(checkpoint_dir=str(tmp_path / "ckpt"), use_class_weights=False)
        trainer = ContinualTrainer(_model(), cfg)

        assert trainer._class_weights(buf) is None


class TestDegeneracyVeto:
    """Regression coverage for the other half of the same real bug: the
    loss-regression gate alone doesn't catch a challenger that has
    collapsed to one action, because under label imbalance a collapsed
    model's loss can look "not meaningfully worse" than the champion's by
    chance. Verified on the real 3000-bar run: 13/13 retrains were
    promoted, and most individual versions called the SAME action on
    100% of their predictions."""

    def test_flags_a_collapsed_prediction_set(self):
        from src.training.trainer import ContinualTrainer

        predicted = np.zeros(200, dtype=np.int64)  # "sell" every single time
        reason = ContinualTrainer._degenerate_prediction_reason(predicted, max_single_class_frac=0.97)
        assert reason is not None
        assert "action 0" in reason
        assert "100.0%" in reason

    def test_passes_a_genuinely_mixed_prediction_set(self):
        from src.training.trainer import ContinualTrainer

        rng = np.random.default_rng(0)
        predicted = rng.integers(0, 3, size=200)  # roughly even three-way split
        reason = ContinualTrainer._degenerate_prediction_reason(predicted, max_single_class_frac=0.97)
        assert reason is None

    def test_empty_predictions_do_not_crash(self):
        from src.training.trainer import ContinualTrainer

        assert ContinualTrainer._degenerate_prediction_reason(np.array([], dtype=np.int64), 0.97) is None

    def test_maybe_retrain_vetoes_promotion_when_challenger_collapses_to_one_action(self, tmp_path):
        """End-to-end reproduction via the real maybe_retrain() path, not
        just the gate function in isolation. Feature windows are all
        identical (all-zero) while labels are skewed but NOT single-class
        (90% sell / 5% hold / 5% buy) -- structurally, the network has
        zero information to tell any two samples apart, so after training
        it is mathematically guaranteed to emit the exact same prediction
        for every validation sample. That's the same observable signature
        as the real collapse found tonight (one action on effectively
        100% of calls), reproduced deterministically instead of relying
        on the stochastic online loop."""
        from src.training.trainer import ContinualTrainer, TrainerConfig

        torch.manual_seed(0)
        np.random.seed(0)
        n = 300
        y = [0] * 270 + [1] * 15 + [2] * 15
        buf = _filled_buffer(tmp_path, y)
        cfg = TrainerConfig(
            checkpoint_dir=str(tmp_path / "ckpt"), min_buffer_size=100, retrain_every_n_new=1,
            batch_size=32, epochs_per_retrain=2, val_fraction=0.2,
        )
        trainer = ContinualTrainer(_model(), cfg)
        trainer.notify_new_samples(1)

        record = trainer.maybe_retrain(buf)

        assert record is not None
        assert record["degeneracy_veto_reason"] is not None
        assert record["promoted"] is False
        # The champion must be untouched -- a veto is a no-op, not a
        # partial promotion.
        assert trainer.model_version() == "v0"

    def test_maybe_retrain_does_not_veto_a_genuinely_learnable_split(self, tmp_path):
        """Sanity check in the other direction: the veto must not fire
        just because it exists -- a challenger that CAN discriminate
        (features actually correlate with the label here) should train
        and get evaluated normally, producing a mixed prediction set."""
        from src.training.trainer import ContinualTrainer, TrainerConfig

        torch.manual_seed(0)
        np.random.seed(0)
        n = 300
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, size=n)
        # Feature windows whose mean level cleanly encodes the label, so
        # the network has real signal to learn from (unlike the all-zero
        # scenario above).
        X = np.zeros((n, WINDOW, N_FEATURES), dtype=np.float32)
        for i, label in enumerate(y):
            X[i, :, 0] = float(label) * 5.0 - 5.0  # sell~-5, hold~0, buy~5
        buf = _filled_buffer(tmp_path, y, X=X)
        cfg = TrainerConfig(
            checkpoint_dir=str(tmp_path / "ckpt"), min_buffer_size=100, retrain_every_n_new=1,
            batch_size=32, epochs_per_retrain=5, val_fraction=0.2,
        )
        trainer = ContinualTrainer(_model(), cfg)
        trainer.notify_new_samples(1)

        record = trainer.maybe_retrain(buf)

        assert record is not None
        assert record["degeneracy_veto_reason"] is None


class TestSeededReproducibility:
    """Regression coverage for the reproducibility bug found while
    building the backtest harness (task #16): maybe_retrain() used to
    create a brand-new `np.random.default_rng()` (OS-entropy-seeded, not
    affected by any global seeding) on every single call, so minibatch
    sampling during a retrain could never be made reproducible no
    matter what the caller did. TrainerConfig.seed -> one persistent
    self._rng fixes that; this proves it end-to-end: two trainers with
    the same seed, fed the identical buffer, must retrain to the exact
    same resulting weights and losses."""

    def _learnable_buffer(self, tmp_path, n=300):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, size=n)
        X = np.zeros((n, WINDOW, N_FEATURES), dtype=np.float32)
        for i, label in enumerate(y):
            X[i, :, 0] = float(label) * 5.0 - 5.0
        return _filled_buffer(tmp_path / "buf", y, X=X)

    def test_same_seed_produces_identical_retrain_outcomes(self, tmp_path):
        from src.utils import seed_everything

        buf = self._learnable_buffer(tmp_path)

        seed_everything(42)
        cfg_a = TrainerConfig(checkpoint_dir=str(tmp_path / "ckpt_a"), min_buffer_size=100,
                               retrain_every_n_new=1, batch_size=32, epochs_per_retrain=3,
                               val_fraction=0.2, seed=7)
        trainer_a = ContinualTrainer(_model(), cfg_a)
        trainer_a.notify_new_samples(1)
        record_a = trainer_a.maybe_retrain(buf)

        seed_everything(42)
        cfg_b = TrainerConfig(checkpoint_dir=str(tmp_path / "ckpt_b"), min_buffer_size=100,
                               retrain_every_n_new=1, batch_size=32, epochs_per_retrain=3,
                               val_fraction=0.2, seed=7)
        trainer_b = ContinualTrainer(_model(), cfg_b)
        trainer_b.notify_new_samples(1)
        record_b = trainer_b.maybe_retrain(buf)

        assert record_a is not None and record_b is not None
        assert record_a["promoted"] == record_b["promoted"]
        assert record_a["challenger_val"]["ce"] == pytest.approx(record_b["challenger_val"]["ce"])
        assert record_a["challenger_val"]["mse"] == pytest.approx(record_b["challenger_val"]["mse"])
        for p_a, p_b in zip(trainer_a.model.parameters(), trainer_b.model.parameters()):
            assert torch.equal(p_a, p_b)

    def test_default_unseeded_config_still_works(self, tmp_path):
        """cfg.seed defaults to None -- maybe_retrain() must still run
        fine (just non-reproducibly), so every pre-existing caller that
        never heard of this option keeps working unchanged."""
        buf = self._learnable_buffer(tmp_path)
        cfg = TrainerConfig(checkpoint_dir=str(tmp_path / "ckpt"), min_buffer_size=100,
                             retrain_every_n_new=1, batch_size=32, epochs_per_retrain=2,
                             val_fraction=0.2)
        assert cfg.seed is None
        trainer = ContinualTrainer(_model(), cfg)
        trainer.notify_new_samples(1)
        record = trainer.maybe_retrain(buf)
        assert record is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
