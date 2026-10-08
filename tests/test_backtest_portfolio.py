import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest

from backtest_portfolio import collapse_trip_ranges, install_kill_switch_auto_reset
from src.data.feed import SyntheticFeed
from src.data.features import LabelConfig
from src.execution.broker import PaperBroker
from src.model.network import ModelConfig, TradingNet
from src.monitor.logger import DecisionLogger
from src.orchestrator import Orchestrator
from src.risk.manager import RiskConfig, RiskManager
from src.training.drift import DriftConfig, DriftMonitor
from src.training.trainer import ContinualTrainer, TrainerConfig

N_FEATURES = 15


def _make_orchestrator(tmp_path, window=20):
    """Mirrors tests/test_orchestrator.py's helper of the same name --
    duplicated rather than imported to keep this file runnable on its
    own and because scripts/ isn't a real package other tests import
    from."""
    tickers = ["T1"]
    feed = SyntheticFeed(tickers, n_bars=500, seed=1)
    model = TradingNet(ModelConfig(n_features=N_FEATURES, window=window, hidden_size=8, trunk_size=8))
    trainer = ContinualTrainer(model, TrainerConfig(checkpoint_dir=str(tmp_path / "checkpoints")))
    risk_manager = RiskManager(RiskConfig())
    broker = PaperBroker()
    logger = DecisionLogger(tmp_path / "decisions.jsonl")
    drift_monitor = DriftMonitor(DriftConfig())
    return Orchestrator(
        tickers=tickers, feed=feed, model=model, trainer=trainer,
        risk_manager=risk_manager, broker=broker, logger=logger,
        drift_monitor=drift_monitor, window=window, label_cfg=LabelConfig(),
        warmup_bars=window + 35,
    )


class TestInstallKillSwitchAutoReset:
    """Regression/behavior coverage for the --ignore-kill-switch escape
    hatch: a real backtest run (task #16) found that DriftMonitor's
    hit-rate halt can trip during a model's cold-start phase and, since
    reset_kill_switch() deliberately requires a human
    (src/risk/manager.py), never recovers for the rest of a long
    backtest -- making it report ~0 trades regardless of how the
    strategy performs afterward. This helper exists to let a backtest
    see past that, strictly as an opt-in analysis tool."""

    def test_default_behavior_is_unchanged_when_not_installed(self, tmp_path):
        """Baseline: without calling install_kill_switch_auto_reset at
        all, a trip must behave exactly as it does in run_paper_trading.py/
        run_live_alpaca.py -- stay engaged, requiring human_confirmed."""
        orch = _make_orchestrator(tmp_path)
        orch.risk_manager.trip_kill_switch("test halt")
        assert orch.risk_manager.kill_switch_engaged()

    def test_installed_trip_is_recorded_and_immediately_reversed(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        trips = install_kill_switch_auto_reset(orch)
        orch.equity_curve.extend([100_000.0, 99_000.0, 95_000.0])  # pretend 3 bars have run

        orch.risk_manager.trip_kill_switch("hit-rate 0.00% below floor 34.00%")

        assert trips == [{"bar": 3, "reason": "hit-rate 0.00% below floor 34.00%"}]
        # The whole point: trading can continue on the very next bar,
        # unlike the unpatched behavior above.
        assert not orch.risk_manager.kill_switch_engaged()

    def test_records_the_correct_bar_for_each_of_several_trips(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        trips = install_kill_switch_auto_reset(orch)

        orch.equity_curve.extend([100_000.0] * 5)
        orch.risk_manager.trip_kill_switch("first halt")
        orch.equity_curve.extend([100_000.0] * 7)  # now at 12
        orch.risk_manager.trip_kill_switch("second halt")

        assert [t["bar"] for t in trips] == [5, 12]
        assert [t["reason"] for t in trips] == ["first halt", "second halt"]
        assert not orch.risk_manager.kill_switch_engaged()

    def test_does_not_touch_risk_manager_or_orchestrator_source(self, tmp_path):
        """Sanity check that this is a pure monkeypatch of one bound
        method on an object the caller owns -- trip_kill_switch on a
        FRESH, un-patched RiskManager instance must still behave with
        the original, unmodified semantics."""
        fresh = RiskManager(RiskConfig())
        fresh.trip_kill_switch("halt")
        assert fresh.kill_switch_engaged()
        with pytest.raises(PermissionError):
            fresh.reset_kill_switch(human_confirmed=False)


class TestCollapseTripRanges:
    """A real 1500-bar backtest produced 1176 individual trips (the halt
    condition persisted bar-after-bar once it fired) -- printing one
    line per trip was unreadable. These are hand-computed against that
    grouping logic directly."""

    def test_hand_computed_grouping(self):
        trips = [{"bar": 5, "reason": "a"}, {"bar": 6, "reason": "b"}, {"bar": 9, "reason": "c"}]
        assert collapse_trip_ranges(trips) == [
            {"start_bar": 5, "end_bar": 6, "start_reason": "a", "end_reason": "b", "n_trips": 2},
            {"start_bar": 9, "end_bar": 9, "start_reason": "c", "end_reason": "c", "n_trips": 1},
        ]

    def test_empty_input_is_empty_output(self):
        assert collapse_trip_ranges([]) == []

    def test_all_consecutive_is_a_single_range(self):
        trips = [{"bar": b, "reason": "x"} for b in range(10, 20)]
        ranges = collapse_trip_ranges(trips)
        assert len(ranges) == 1
        assert ranges[0]["start_bar"] == 10
        assert ranges[0]["end_bar"] == 19
        assert ranges[0]["n_trips"] == 10

    def test_no_two_trips_adjacent_is_all_singleton_ranges(self):
        trips = [{"bar": 1, "reason": "a"}, {"bar": 5, "reason": "b"}, {"bar": 100, "reason": "c"}]
        ranges = collapse_trip_ranges(trips)
        assert len(ranges) == 3
        assert all(r["n_trips"] == 1 for r in ranges)
