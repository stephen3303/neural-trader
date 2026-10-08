import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.drift import DriftConfig, DriftMonitor


def test_state_survives_a_save_load_round_trip(tmp_path):
    """Regression coverage: before save_state()/load_state() existed, a
    process restart silently reset hit-rate/calibration/drawdown tracking
    to empty, meaning should_halt() would stay quiet for cfg.min_samples
    worth of fresh data after every restart -- exactly when a system that
    just crashed mid-session most needs it to still be watching."""
    path = tmp_path / "drift_state.json"
    dm = DriftMonitor(DriftConfig(window=50, min_samples=5))
    for i in range(10):
        dm.record_prediction_outcome(correct=(i % 3 != 0), confidence=0.6 + i * 0.01)
    for equity in [100_000.0, 101_000.0, 99_000.0, 98_500.0]:
        dm.record_equity(equity)
    before = dm.snapshot()
    dm.save_state(path)

    fresh = DriftMonitor(DriftConfig(window=50, min_samples=5))
    assert fresh.snapshot()["n_outcomes"] == 0  # before loading: empty, as if nothing happened
    loaded = fresh.load_state(path)

    assert loaded is True
    after = fresh.snapshot()
    assert after == before
    assert fresh._peak_equity == dm._peak_equity


def test_load_state_returns_false_with_no_file(tmp_path):
    dm = DriftMonitor(DriftConfig())
    assert dm.load_state(tmp_path / "does-not-exist.json") is False
    assert dm.snapshot()["n_outcomes"] == 0


def test_loaded_state_still_respects_this_instances_maxlen(tmp_path):
    """If cfg.window shrank since the state was saved (e.g. a config
    change between restarts), the restored deque must still honor the
    NEW instance's maxlen rather than silently growing unbounded."""
    path = tmp_path / "drift_state.json"
    wide = DriftMonitor(DriftConfig(window=100, min_samples=1))
    for i in range(20):
        wide.record_prediction_outcome(correct=True, confidence=0.5)
    wide.save_state(path)

    narrow = DriftMonitor(DriftConfig(window=5, min_samples=1))
    narrow.load_state(path)

    assert len(narrow._outcomes) == 5  # capped by the NEW instance's maxlen, not the saved 20
