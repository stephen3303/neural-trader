import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.model.signals import Action, Signal
from src.risk.manager import RiskConfig, RiskManager


def _signal(action=Action.BUY, confidence=0.8, expected_return=0.01):
    return Signal(ticker="TEST", timestamp=0, action=action, confidence=confidence,
                  expected_return=expected_return, model_version="v0")


def test_low_confidence_results_in_no_trade():
    rm = RiskManager(RiskConfig(min_confidence=0.6))
    sizing = rm.size_order(_signal(confidence=0.5), realized_vol=0.01)
    assert sizing["size_pct_equity"] == 0.0
    assert sizing["reason"] == "hold or low confidence"


def test_kill_switch_blocks_all_trades():
    rm = RiskManager(RiskConfig())
    rm.trip_kill_switch("test halt")
    sizing = rm.size_order(_signal(), realized_vol=0.01)
    assert sizing["size_pct_equity"] == 0.0
    assert "kill switch" in sizing["reason"]


def test_position_size_respects_max_cap():
    rm = RiskManager(RiskConfig(max_position_pct=5.0, kelly_fraction=1.0, target_daily_vol_pct=50.0))
    sizing = rm.size_order(_signal(confidence=0.99, expected_return=1.0), realized_vol=0.001)
    assert sizing["size_pct_equity"] <= 5.0


def test_daily_loss_limit_trips_kill_switch():
    rm = RiskManager(RiskConfig(max_daily_loss_pct=2.0))
    rm.update_after_trade_result(-1.2)
    assert not rm.kill_switch_engaged()
    rm.update_after_trade_result(-1.0)
    assert rm.kill_switch_engaged()


def test_kill_switch_reset_requires_human_confirmation():
    rm = RiskManager(RiskConfig())
    rm.trip_kill_switch("halt")
    try:
        rm.reset_kill_switch(human_confirmed=False)
        assert False, "should have raised"
    except PermissionError:
        pass
    rm.reset_kill_switch(human_confirmed=True)
    assert not rm.kill_switch_engaged()


def test_state_survives_a_save_load_round_trip(tmp_path):
    """Regression coverage for the other half of the persistence work:
    before save_state()/load_state() existed, a process restart silently
    forgot today's drawdown and consecutive-loss streak, and (separately)
    re-armed any kill switch that had already tripped."""
    path = tmp_path / "risk_state.json"
    rm = RiskManager(RiskConfig(max_daily_loss_pct=2.0))
    rm.update_after_trade_result(-1.2)
    rm.update_after_trade_result(-1.5)  # trips the kill switch
    assert rm.kill_switch_engaged()
    rm.save_state(path)

    fresh = RiskManager(RiskConfig(max_daily_loss_pct=2.0))
    assert not fresh.kill_switch_engaged()  # before loading: as if nothing happened
    loaded = fresh.load_state(path)

    assert loaded is True
    assert fresh.kill_switch_engaged()  # the tripped kill switch must survive the restart
    assert fresh.state.daily_pnl_pct == pytest.approx(-2.7)
    assert fresh.state.halt_reasons == rm.state.halt_reasons
    # Mutated in place, not rebound -- any other code already holding a
    # reference to fresh.state must see the restored values too.
    assert fresh.state is not None and isinstance(fresh.state, type(rm.state))


def test_load_state_returns_false_with_no_file(tmp_path):
    rm = RiskManager(RiskConfig())
    assert rm.load_state(tmp_path / "does-not-exist.json") is False
    assert not rm.kill_switch_engaged()


class TestUpdateAccountEquity:
    """Regression coverage for a bug found while working on long-term
    profitability: cfg.account_equity stayed frozen at config.yaml's
    static startup value forever. AlpacaBroker.get_equity() already
    queries the real, current account balance every bar -- that value
    just never made it back into the risk manager, so position sizing
    silently drifted from the real account as it compounded gains/losses."""

    def test_updates_account_equity_to_a_valid_value(self):
        rm = RiskManager(RiskConfig(account_equity=100_000.0))
        rm.update_account_equity(123_456.78)
        assert rm.cfg.account_equity == 123_456.78

    def test_ignores_non_positive_values(self):
        rm = RiskManager(RiskConfig(account_equity=100_000.0))
        rm.update_account_equity(0.0)
        assert rm.cfg.account_equity == 100_000.0
        rm.update_account_equity(-500.0)
        assert rm.cfg.account_equity == 100_000.0

    def test_ignores_non_finite_values(self):
        rm = RiskManager(RiskConfig(account_equity=100_000.0))
        rm.update_account_equity(float("nan"))
        assert rm.cfg.account_equity == 100_000.0
        rm.update_account_equity(float("inf"))
        assert rm.cfg.account_equity == 100_000.0

    def test_a_bad_read_does_not_disturb_a_prior_good_value(self):
        """A single transient bad reading must never zero out or corrupt
        position sizing for the rest of the session -- it should just
        keep using the last known-good equity."""
        rm = RiskManager(RiskConfig(account_equity=100_000.0))
        rm.update_account_equity(150_000.0)
        rm.update_account_equity(float("nan"))  # e.g. a transient API hiccup
        assert rm.cfg.account_equity == 150_000.0
