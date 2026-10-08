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


class TestExternalResetRequest:
    """Regression coverage for the dashboard's "Reset kill switch"
    button: scripts/serve_dashboard.py runs as a separate process from
    the live trading loop and has no reference to its RiskManager, so
    it can only ever leave a request file lying around
    (write_reset_request) for the trading loop to notice and act on
    itself (check_for_reset_request), on its own next bar."""

    def test_a_pending_request_resets_an_engaged_kill_switch(self, tmp_path):
        rm = RiskManager(RiskConfig())
        rm.trip_kill_switch("daily loss breached")
        RiskManager.write_reset_request(tmp_path, source="dashboard")

        result = rm.check_for_reset_request(tmp_path)

        assert not rm.kill_switch_engaged()
        assert result == {"source": "dashboard", "cleared_reasons": ["daily loss breached"]}

    def test_the_request_file_is_consumed_so_it_cannot_fire_twice(self, tmp_path):
        rm = RiskManager(RiskConfig())
        rm.trip_kill_switch("halt")
        RiskManager.write_reset_request(tmp_path, source="dashboard")

        rm.check_for_reset_request(tmp_path)
        rm.trip_kill_switch("halt again")
        second = rm.check_for_reset_request(tmp_path)

        assert second is None
        assert rm.kill_switch_engaged()  # the second halt is untouched

    def test_no_request_file_is_a_no_op(self, tmp_path):
        rm = RiskManager(RiskConfig())
        rm.trip_kill_switch("halt")

        assert rm.check_for_reset_request(tmp_path) is None
        assert rm.kill_switch_engaged()

    def test_a_request_that_arrives_while_already_armed_is_a_silent_no_op(self, tmp_path):
        # Not halted to begin with -- a stray/duplicate/late request must
        # not report a reset that never happened, and must still consume
        # (delete) the sentinel file so it can't resurface later.
        rm = RiskManager(RiskConfig())
        RiskManager.write_reset_request(tmp_path, source="dashboard")

        result = rm.check_for_reset_request(tmp_path)

        assert result is None
        assert not rm.kill_switch_engaged()
        assert not (tmp_path / RiskManager.RESET_REQUEST_FILENAME).exists()

    def test_a_corrupt_request_file_still_resets_rather_than_raising(self, tmp_path):
        rm = RiskManager(RiskConfig())
        rm.trip_kill_switch("halt")
        (tmp_path / RiskManager.RESET_REQUEST_FILENAME).write_text("not json")

        result = rm.check_for_reset_request(tmp_path)

        assert not rm.kill_switch_engaged()
        assert result["source"] == "dashboard"  # falls back to the default


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


class TestUpdateOpenExposure:
    """Regression coverage for a second bug found alongside the account-
    equity one: RiskState.open_notional_pct -- the number size_order()'s
    portfolio-wide gross-exposure cap (cfg.max_gross_exposure_pct) checks
    against -- was initialized to 0.0 and never updated ANYWHERE in the
    codebase (confirmed by grep -- not even a test exercised a nonzero
    value), making that cap completely non-functional: the check
    `open_notional_pct >= max_gross_exposure_pct` could never fire since
    0.0 is never >= a positive cap, and every size_order() call computed
    headroom as the FULL configured cap, forever. With enough tickers
    each independently sized up to max_position_pct, aggregate exposure
    across the whole portfolio could exceed the configured limit with
    nothing to stop it -- verified directly on the real pipeline (8
    tickers, max_position_pct=20%, max_gross_exposure_pct=25%): before
    this fix, open_notional_pct stayed exactly 0.0 for the entire run
    regardless of how many trades filled (271 trades, zero exposure
    tracking); after it, the gross-exposure cap visibly constrains
    sizing as real positions accumulate."""

    def test_updates_open_exposure_to_a_valid_value(self):
        rm = RiskManager(RiskConfig())
        rm.update_open_exposure(42.5)
        assert rm.state.open_notional_pct == 42.5

    def test_the_cap_now_actually_blocks_new_orders_once_breached(self):
        """The real end-to-end effect of the fix: size_order() must
        refuse new orders once open_notional_pct reaches the configured
        cap -- this was impossible to exercise before, since nothing
        could ever make open_notional_pct nonzero."""
        rm = RiskManager(RiskConfig(max_gross_exposure_pct=10.0))
        rm.update_open_exposure(10.0)  # already at the cap
        sizing = rm.size_order(_signal(), realized_vol=0.01)
        assert sizing["size_pct_equity"] == 0.0
        assert sizing["reason"] == "max gross exposure reached"

    def test_sizing_is_capped_to_remaining_headroom_not_the_full_cap(self):
        rm = RiskManager(RiskConfig(max_gross_exposure_pct=10.0, max_position_pct=50.0,
                                     kelly_fraction=1.0, target_daily_vol_pct=50.0))
        rm.update_open_exposure(7.0)  # 7 of 10 points of headroom already used
        sizing = rm.size_order(_signal(confidence=0.99, expected_return=1.0), realized_vol=0.001)
        assert sizing["size_pct_equity"] <= 3.0 + 1e-6  # only 3 points of headroom left

    def test_ignores_negative_values(self):
        rm = RiskManager(RiskConfig())
        rm.update_open_exposure(15.0)
        rm.update_open_exposure(-5.0)
        assert rm.state.open_notional_pct == 15.0

    def test_ignores_non_finite_values(self):
        rm = RiskManager(RiskConfig())
        rm.update_open_exposure(15.0)
        rm.update_open_exposure(float("nan"))
        assert rm.state.open_notional_pct == 15.0
        rm.update_open_exposure(float("inf"))
        assert rm.state.open_notional_pct == 15.0

    def test_zero_is_a_valid_reading_not_ignored(self):
        """Unlike update_account_equity (where 0.0 is nonsensical for an
        account balance), 0.0 open exposure is a perfectly normal,
        common state (no open positions) and must be accepted, not
        treated as a bad reading."""
        rm = RiskManager(RiskConfig())
        rm.update_open_exposure(15.0)
        rm.update_open_exposure(0.0)
        assert rm.state.open_notional_pct == 0.0


class TestUpdatePerTickerExposure:
    """Regression coverage for a bug found by extending the dead-code
    audit one step further: max_position_pct ("single position cap, %
    of equity") was only ever checked against each INDIVIDUAL new
    order's own size -- never against how much of that ticker was
    already held. A sustained run of same-direction signals on one
    ticker (a real trend, which is exactly the condition this strategy
    is built to ride) could keep adding to that ticker's position bar
    after bar with nothing to stop it. Verified directly on a real
    3,000-bar/12-ticker run: before this fix, QQQ's position reached
    17.9% of equity and NVDA's 13.9%, both past the configured 10% cap,
    with every individual order along the way still correctly <= 10% on
    its own; after it, the worst observed was ~10.5% (the same kind of
    brief mark-to-market overshoot documented on the portfolio-wide cap
    -- the cap bounds NEW orders at sizing time, it can't retroactively
    shrink an already-open position that then drifts)."""

    def test_updates_a_tickers_exposure_to_a_valid_value(self):
        rm = RiskManager(RiskConfig())
        rm.update_per_ticker_exposure("AAPL", 7.5)
        assert rm.state.per_ticker_notional_pct["AAPL"] == 7.5

    def test_tracks_multiple_tickers_independently(self):
        rm = RiskManager(RiskConfig())
        rm.update_per_ticker_exposure("AAPL", 7.5)
        rm.update_per_ticker_exposure("MSFT", 2.0)
        assert rm.state.per_ticker_notional_pct == {"AAPL": 7.5, "MSFT": 2.0}

    def test_the_cap_now_actually_blocks_a_new_order_once_a_tickers_position_is_at_the_cap(self):
        """The real end-to-end effect: size_order() must refuse a new
        order for a ticker that's already at max_position_pct, even
        though nothing is wrong with the order itself or with the
        portfolio-wide gross exposure -- this was impossible to exercise
        before, since nothing could ever make per_ticker_notional_pct
        nonzero."""
        rm = RiskManager(RiskConfig(max_position_pct=10.0))
        rm.update_per_ticker_exposure("TEST", 10.0)  # already at this ticker's cap
        sizing = rm.size_order(_signal(), realized_vol=0.01)
        assert sizing["size_pct_equity"] == 0.0
        assert sizing["reason"] == "max position pct reached for this ticker"

    def test_sizing_is_capped_to_the_tickers_remaining_headroom_not_the_full_cap(self):
        rm = RiskManager(RiskConfig(max_position_pct=10.0, max_gross_exposure_pct=100.0,
                                     kelly_fraction=1.0, target_daily_vol_pct=50.0))
        rm.update_per_ticker_exposure("TEST", 6.0)  # 6 of 10 points already used on this ticker
        sizing = rm.size_order(_signal(confidence=0.99, expected_return=1.0), realized_vol=0.001)
        assert sizing["size_pct_equity"] <= 4.0 + 1e-6  # only 4 points of headroom left

    def test_a_different_tickers_exposure_does_not_affect_this_ones_sizing(self):
        rm = RiskManager(RiskConfig(max_position_pct=10.0))
        rm.update_per_ticker_exposure("OTHER", 10.0)  # OTHER is maxed out
        sizing = rm.size_order(_signal(), realized_vol=0.01)  # signal is for "TEST"
        assert sizing["size_pct_equity"] > 0.0  # TEST itself has no recorded exposure

    def test_ignores_negative_values(self):
        rm = RiskManager(RiskConfig())
        rm.update_per_ticker_exposure("AAPL", 7.5)
        rm.update_per_ticker_exposure("AAPL", -2.0)
        assert rm.state.per_ticker_notional_pct["AAPL"] == 7.5

    def test_ignores_non_finite_values(self):
        rm = RiskManager(RiskConfig())
        rm.update_per_ticker_exposure("AAPL", 7.5)
        rm.update_per_ticker_exposure("AAPL", float("nan"))
        assert rm.state.per_ticker_notional_pct["AAPL"] == 7.5
        rm.update_per_ticker_exposure("AAPL", float("inf"))
        assert rm.state.per_ticker_notional_pct["AAPL"] == 7.5

    def test_zero_is_a_valid_reading_not_ignored(self):
        rm = RiskManager(RiskConfig())
        rm.update_per_ticker_exposure("AAPL", 7.5)
        rm.update_per_ticker_exposure("AAPL", 0.0)
        assert rm.state.per_ticker_notional_pct["AAPL"] == 0.0

    def test_state_survives_a_save_load_round_trip(self, tmp_path):
        rm = RiskManager(RiskConfig())
        rm.update_per_ticker_exposure("AAPL", 7.5)
        rm.update_per_ticker_exposure("MSFT", 2.0)
        path = tmp_path / "risk_state.json"
        rm.save_state(path)

        fresh = RiskManager(RiskConfig())
        assert fresh.load_state(path) is True
        assert fresh.state.per_ticker_notional_pct == {"AAPL": 7.5, "MSFT": 2.0}

    def test_loading_a_state_file_saved_before_this_field_existed_does_not_raise(self, tmp_path):
        """Backward compatibility: a risk_state.json written before
        per_ticker_notional_pct existed has no such key at all -- resuming
        from it must not KeyError."""
        import json
        path = tmp_path / "old_risk_state.json"
        path.write_text(json.dumps({
            "trading_enabled": True, "halt_reasons": [], "daily_pnl_pct": 0.0,
            "consecutive_losses": 0, "open_notional_pct": 0.0,
        }))
        rm = RiskManager(RiskConfig())
        assert rm.load_state(path) is True
        assert rm.state.per_ticker_notional_pct == {}


class TestStopLossVolScaling:
    """Regression coverage for replacing one flat hard_stop_loss_pct
    (applied identically to every ticker) with a per-ticker,
    volatility-scaled stop -- see stop_loss_pct_for()'s own docstring
    for the concrete numbers motivating this (a flat 3% stop was
    routinely triggered by ordinary noise on a choppy ticker, while
    being needlessly loose on a calm one -- exactly the "too many
    exits on volatile names" problem this closes)."""

    def test_scales_linearly_with_realized_vol(self):
        rm = RiskManager(RiskConfig(stop_loss_vol_multiplier=3.0,
                                     min_stop_loss_pct=0.0, max_stop_loss_pct=100.0))
        # 1% per-bar vol * 3.0 multiplier = 3% stop.
        assert rm.stop_loss_pct_for(0.01) == pytest.approx(3.0)
        # Double the vol, double the stop -- same multiplier.
        assert rm.stop_loss_pct_for(0.02) == pytest.approx(6.0)

    def test_a_calmer_ticker_gets_a_tighter_stop_than_a_choppier_one(self):
        rm = RiskManager(RiskConfig(stop_loss_vol_multiplier=3.0,
                                     min_stop_loss_pct=0.0, max_stop_loss_pct=100.0))
        calm = rm.stop_loss_pct_for(0.003)    # a quiet ticker's typical per-bar vol
        choppy = rm.stop_loss_pct_for(0.015)  # a volatile ticker's
        assert calm < choppy

    def test_clamped_to_the_floor_for_an_ultra_calm_ticker(self):
        rm = RiskManager(RiskConfig(stop_loss_vol_multiplier=3.0,
                                     min_stop_loss_pct=1.5, max_stop_loss_pct=8.0))
        # 0.1% vol * 3.0 = 0.3%, well under the 1.5% floor.
        assert rm.stop_loss_pct_for(0.001) == pytest.approx(1.5)

    def test_clamped_to_the_ceiling_for_a_genuinely_wild_ticker(self):
        rm = RiskManager(RiskConfig(stop_loss_vol_multiplier=3.0,
                                     min_stop_loss_pct=1.5, max_stop_loss_pct=8.0))
        # 5% vol * 3.0 = 15%, well over the 8% ceiling.
        assert rm.stop_loss_pct_for(0.05) == pytest.approx(8.0)

    def test_non_finite_or_non_positive_vol_falls_back_to_the_floor(self):
        rm = RiskManager(RiskConfig(min_stop_loss_pct=1.5))
        assert rm.stop_loss_pct_for(float("nan")) == 1.5
        assert rm.stop_loss_pct_for(float("inf")) == 1.5
        assert rm.stop_loss_pct_for(0.0) == 1.5
        assert rm.stop_loss_pct_for(-0.01) == 1.5

    def test_size_order_returns_the_vol_scaled_stop_not_a_flat_constant(self):
        rm = RiskManager(RiskConfig(stop_loss_vol_multiplier=3.0,
                                     min_stop_loss_pct=0.0, max_stop_loss_pct=100.0))
        low_vol_sizing = rm.size_order(_signal(), realized_vol=0.003)
        high_vol_sizing = rm.size_order(_signal(), realized_vol=0.015)
        assert low_vol_sizing["stop_loss_pct"] == pytest.approx(rm.stop_loss_pct_for(0.003))
        assert high_vol_sizing["stop_loss_pct"] == pytest.approx(rm.stop_loss_pct_for(0.015))
        assert low_vol_sizing["stop_loss_pct"] < high_vol_sizing["stop_loss_pct"]

    def test_a_no_trade_decision_still_reports_stop_loss_pct_as_none(self):
        """_no_trade's stop_loss_pct stays None regardless of vol --
        there's no real position, so there's nothing to attach a stop
        to. Orchestrator relies on this: it stores sizing["stop_loss_pct"]
        on every pending entry unconditionally, and _check_stop_losses
        already skips any entry with size_pct_equity <= 0 before ever
        reading it."""
        rm = RiskManager(RiskConfig(min_confidence=0.9))
        sizing = rm.size_order(_signal(confidence=0.1), realized_vol=0.05)
        assert sizing["size_pct_equity"] == 0.0
        assert sizing["stop_loss_pct"] is None
