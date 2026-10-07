import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
