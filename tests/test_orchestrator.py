import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.feed import SyntheticFeed
from src.data.features import LabelConfig
from src.execution.broker import PaperBroker, Position
from src.model.network import ModelConfig, TradingNet
from src.monitor.logger import DecisionLogger
from src.orchestrator import Orchestrator
from src.risk.manager import RiskConfig, RiskManager
from src.training.drift import DriftConfig, DriftMonitor
from src.training.trainer import ContinualTrainer, TrainerConfig

N_FEATURES = 15


def _make_orchestrator(tmp_path, window=20):
    tickers = ["T1"]
    feed = SyntheticFeed(tickers, n_bars=500, seed=1)
    model = TradingNet(ModelConfig(n_features=N_FEATURES, window=window, hidden_size=8, trunk_size=8))
    trainer = ContinualTrainer(model, TrainerConfig(checkpoint_dir=str(tmp_path / "checkpoints")))
    risk_manager = RiskManager(RiskConfig())
    broker = PaperBroker()
    logger = DecisionLogger(tmp_path / "decisions.jsonl")
    drift_monitor = DriftMonitor(DriftConfig())
    orch = Orchestrator(
        tickers=tickers, feed=feed, model=model, trainer=trainer,
        risk_manager=risk_manager, broker=broker, logger=logger,
        drift_monitor=drift_monitor, window=window, label_cfg=LabelConfig(),
        # features.py's rolling indicators (sma_30_dev, vol_z, etc.) need up
        # to 30 bars of their own warm-up before they stop being NaN, on top
        # of the `window`-length lookback build_windows() needs -- too little
        # history here means build_windows() filters every candidate window
        # out for containing NaNs and _try_predict() returns None.
        warmup_bars=window + 35,
    )
    return orch


def test_predictions_use_the_most_recently_promoted_model(tmp_path):
    """Regression test for a real bug found while reviewing the continual-
    retraining path: ContinualTrainer.maybe_retrain() promotes a challenger
    by REBINDING its own `self.model` attribute to a new object
    (`self.model = challenger`) -- it never mutates the original model
    in place. Orchestrator previously called `self.model.predict(...)`,
    where `self.model` was a separate reference captured once at
    construction time. After the very first promoted retrain, those two
    references point at two different objects: `trainer.model` is the
    improved challenger, but `orchestrator.model` is still the stale,
    pre-promotion model -- so every prediction for the rest of the run
    silently ignores all future retraining, while the logs claim a newer
    `model_version` the whole time. This test forces a promotion directly
    (bypassing the stochastic retrain gate, for a fast deterministic test)
    and asserts a prediction right after it actually reflects the new
    model, not the stale one."""
    orch = _make_orchestrator(tmp_path)

    # Prime enough history to make a real prediction.
    for bar in orch.feed.stream(orch.tickers):
        orch._update_history(bar)
        orch._bar_count[bar.ticker] += 1
        if len(orch._history[bar.ticker]) >= orch.warmup_bars:
            break

    # Simulate a promoted retrain exactly like ContinualTrainer.maybe_retrain
    # does on promotion: swap in a NEW model object, don't mutate the old one.
    old_model = orch.trainer.model
    new_model = TradingNet(old_model.cfg)
    # Make the new model's output unambiguously distinguishable from the old
    # one's, so "which model actually produced this prediction" is observable.
    with torch.no_grad():
        new_model.action_head.bias[:] = torch.tensor([10.0, -10.0, -10.0])  # forces "sell" (index 0)
        old_model.action_head.bias[:] = torch.tensor([-10.0, -10.0, 10.0])  # forces "buy" (index 2)
    orch.trainer.model = new_model
    orch.trainer._version += 1

    pred = orch._try_predict(orch.tickers[0], pd_timestamp_stub())
    assert pred is not None, "expected a prediction once warmup_bars of history exist"
    assert pred["action"] == 0, (
        "prediction used the stale pre-promotion model (predicted 'buy', action=2) "
        "instead of the newly promoted one (should predict 'sell', action=0) -- "
        "Orchestrator is not reading orch.trainer.model for inference"
    )


def pd_timestamp_stub():
    import pandas as pd
    return pd.Timestamp("2024-01-01")


class _FixedCloseBar:
    """Minimal stand-in for a real Bar -- _resolve_matured reads `.close`
    and `.timestamp` off the "current bar" argument."""
    def __init__(self, close: float, timestamp="2024-01-01"):
        self.close = close
        self.timestamp = timestamp


def _enqueue_pending(orch, ticker, *, action, size_pct_equity, entry_price=100.0,
                      mature_at_count=0, filled_quantity=None, stop_loss_pct=3.0):
    if filled_quantity is None:
        # Default to whatever quantity size_pct_equity-worth of notional
        # at entry_price would have filled (ignoring the tiny PaperBroker
        # slippage adjustment -- irrelevant to what these defaults need
        # to support), unless size_pct_equity is 0 (nothing was ever
        # filled, so there is nothing to later close). Keeps every
        # existing test that doesn't care about the real broker close
        # (most of this file) still exercising _resolve_one's new
        # close_quantity call with a realistic quantity, exactly as a
        # real pending entry from run() would carry, rather than
        # silently skipping that code path. Tests that DO care about the
        # resulting broker position (TestRealBrokerCloseOnResolution)
        # pass filled_quantity explicitly instead.
        filled_quantity = (
            (size_pct_equity / 100.0) * orch.risk_manager.cfg.account_equity / entry_price
            if size_pct_equity > 0 else 0.0
        )
    orch._pending[ticker].append({
        "decision_id": "d-test",
        "entry_price": entry_price,
        "action": action,          # 0=sell, 1=hold, 2=buy
        "confidence": 0.9,
        "feature_window": np.zeros((orch.window, N_FEATURES), dtype=np.float32),
        "mature_at_count": mature_at_count,
        "size_pct_equity": size_pct_equity,
        "filled_quantity": filled_quantity,
        "stop_loss_pct": stop_loss_pct,
    })
    orch._bar_count[ticker] = 0


class _OHLCBar:
    """Minimal stand-in for a real Bar carrying intrabar high/low --
    unlike _FixedCloseBar above, _check_stop_losses reads `.high`/`.low`
    (not just `.close`) off the current bar, so it can catch a stop that
    was breached and recovered within the same bar, exactly like a real
    stop order would have filled intrabar."""
    def __init__(self, high, low, close=None, timestamp="2024-01-01"):
        self.high = high
        self.low = low
        self.close = close if close is not None else (high + low) / 2
        self.timestamp = timestamp


class TestRiskManagerReceivesRealizedPnl:
    """Regression coverage for a bug found while working on long-term
    profitability: RiskManager.update_after_trade_result() -- the function
    that drives cfg.max_daily_loss_pct and cfg.max_consecutive_losses --
    was never called anywhere in Orchestrator's main loop. It was only
    ever exercised by tests/test_risk.py's direct unit tests. In real
    trading, that meant the daily-loss and consecutive-losses kill-switch
    conditions could never trip, no matter how badly a session went --
    only DriftMonitor's separate hit-rate/brier/equity-drawdown halt was
    ever actually live."""

    def test_a_filled_losing_trade_updates_the_risk_manager(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        # A BUY sized at 10% of equity, entered at 100.0, that matures at
        # 90.0 -- a 10% adverse move against a long position.
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0, entry_price=100.0)

        assert orch.risk_manager.state.daily_pnl_pct == 0.0
        orch._resolve_matured(ticker, _FixedCloseBar(90.0))

        # 10% size * +1 (buy) * -0.10 (realized_ret) = -1.0 percentage
        # points of account equity -- must be visible on the risk manager,
        # not stuck at the initial 0.0.
        assert orch.risk_manager.state.daily_pnl_pct == pytest.approx(-1.0)
        assert orch.risk_manager.state.consecutive_losses == 1

    def test_a_filled_winning_trade_resets_the_consecutive_loss_streak(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.risk_manager.state.consecutive_losses = 4  # pretend a losing streak is already underway
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0, entry_price=100.0)

        orch._resolve_matured(ticker, _FixedCloseBar(110.0))  # +10% move in favor of the BUY

        assert orch.risk_manager.state.daily_pnl_pct == pytest.approx(1.0)
        assert orch.risk_manager.state.consecutive_losses == 0

    def test_unsized_predictions_do_not_touch_the_risk_manager(self, tmp_path):
        """A hold / zero-confidence / risk-capped prediction has
        size_pct_equity == 0.0 (no real trade happened) and must not be
        counted as a realized win or loss -- in particular it must not
        reset an in-progress consecutive-loss streak."""
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.risk_manager.state.consecutive_losses = 2
        _enqueue_pending(orch, ticker, action=1, size_pct_equity=0.0, entry_price=100.0)

        orch._resolve_matured(ticker, _FixedCloseBar(90.0))  # big move, but nothing was ever sized

        assert orch.risk_manager.state.daily_pnl_pct == 0.0
        assert orch.risk_manager.state.consecutive_losses == 2  # untouched

    def test_repeated_losing_trades_trip_the_kill_switch_through_the_real_loop_path(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.risk_manager.cfg.max_consecutive_losses = 3

        for i in range(3):
            _enqueue_pending(orch, ticker, action=2, size_pct_equity=5.0, entry_price=100.0)
            orch._resolve_matured(ticker, _FixedCloseBar(95.0))  # a loss every time

        assert orch.risk_manager.kill_switch_engaged()
        assert "consecutive losing trades" in orch.risk_manager.state.halt_reasons[-1]

    def test_run_only_records_size_pct_equity_when_an_order_actually_fills(self, tmp_path, monkeypatch):
        """End-to-end check of the actual run() plumbing (not a
        reimplementation of it): force every prediction to be sized, run
        one real bar through the real loop, and confirm the resulting
        pending entry's size_pct_equity matches the sizing decision when
        the broker fills, and is 0.0 when the broker rejects the order --
        exactly what _resolve_matured relies on to gate the risk-manager
        update above."""
        from src.execution.broker import Fill
        from src.model.signals import Action

        fixed_sizing = {"action": Action.BUY, "size_pct_equity": 7.5,
                         "stop_loss_pct": 3.0, "reason": "sized"}

        def make_sizing(signal, vol):
            return {**fixed_sizing, "ticker": signal.ticker, "model_version": signal.model_version}

        def fake_fill(ticker, action, notional, ref_price, ts):
            return Fill(ticker=ticker, action=action, quantity=notional / ref_price,
                        price=ref_price, commission=0.0, timestamp=ts)

        # Case 1: broker fills -> pending entry carries the real size.
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        monkeypatch.setattr(orch.risk_manager, "size_order", make_sizing)
        monkeypatch.setattr(orch.broker, "submit_order", fake_fill)
        orch.run(max_bars=1)
        assert len(orch._pending[ticker]) == 1
        assert orch._pending[ticker][0]["size_pct_equity"] == 7.5

        # Case 2: same sizing decision, but the broker rejects the order
        # (returns None) -- the pending entry's size must be 0.0, so
        # _resolve_matured never counts this as a realized trade.
        orch2 = _make_orchestrator(tmp_path)
        monkeypatch.setattr(orch2.risk_manager, "size_order", make_sizing)
        monkeypatch.setattr(orch2.broker, "submit_order",
                             lambda ticker, action, notional, ref_price, ts: None)
        orch2.run(max_bars=1)
        assert len(orch2._pending[ticker]) == 1
        assert orch2._pending[ticker][0]["size_pct_equity"] == 0.0


class TestStatePersistence:
    """Regression coverage for persisting risk/drift state across
    restarts -- the other half of the known limitation from the
    checkpoint-resume work. Orchestrator writes risk_state_path /
    drift_state_path after every state-changing event when they're set
    (None, the default, means fully in-memory, e.g. for backtests)."""

    def test_a_realized_trade_writes_risk_state_to_disk(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch.risk_state_path = tmp_path / "risk_state.json"
        ticker = orch.tickers[0]
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0, entry_price=100.0)

        assert not orch.risk_state_path.exists()
        orch._resolve_matured(ticker, _FixedCloseBar(90.0))

        assert orch.risk_state_path.exists()
        from src.risk.manager import RiskConfig, RiskManager
        fresh = RiskManager(RiskConfig())
        fresh.load_state(orch.risk_state_path)
        assert fresh.state.daily_pnl_pct == orch.risk_manager.state.daily_pnl_pct

    def test_a_resolved_outcome_writes_drift_state_to_disk(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch.drift_state_path = tmp_path / "drift_state.json"
        ticker = orch.tickers[0]
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=0.0, entry_price=100.0)

        assert not orch.drift_state_path.exists()
        orch._resolve_matured(ticker, _FixedCloseBar(90.0))

        assert orch.drift_state_path.exists()
        from src.training.drift import DriftConfig, DriftMonitor
        fresh = DriftMonitor(DriftConfig())
        fresh.load_state(orch.drift_state_path)
        assert fresh.snapshot()["n_outcomes"] == orch.drift_monitor.snapshot()["n_outcomes"]

    def test_no_state_files_written_when_paths_are_not_set(self, tmp_path):
        """Default behavior (backtests, run_paper_trading.py) must stay
        fully in-memory -- no disk writes at all unless a path is given."""
        orch = _make_orchestrator(tmp_path)
        assert orch.risk_state_path is None
        assert orch.drift_state_path is None
        ticker = orch.tickers[0]
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0, entry_price=100.0)
        orch._resolve_matured(ticker, _FixedCloseBar(90.0))
        assert not (tmp_path / "risk_state.json").exists()
        assert not (tmp_path / "drift_state.json").exists()


class TestAccountEquitySync:
    """Regression coverage for a bug found while working on long-term
    profitability: risk_manager.cfg.account_equity stayed frozen at
    config.yaml's static startup value forever, even though the broker's
    real current equity was already being fetched every bar (just never
    fed back into the risk manager). This matters most for AlpacaBroker,
    whose get_equity() queries the real account balance, but PaperBroker
    benefits too -- sizing should track the account's actual simulated
    equity as it compounds, not a number frozen at whatever it was when
    the process started."""

    def test_run_syncs_account_equity_from_the_broker_before_sizing(self, tmp_path, monkeypatch):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        starting_equity = orch.risk_manager.cfg.account_equity
        # A broker reporting equity very different from config.yaml's
        # static assumption -- simulating an account that's drifted (or,
        # for AlpacaBroker, simply the real balance never matching the
        # static config value in the first place).
        new_equity = starting_equity * 2.5
        monkeypatch.setattr(orch.broker, "get_equity", lambda mark_prices=None: new_equity)

        orch.run(max_bars=1)

        assert orch.risk_manager.cfg.account_equity == new_equity
        assert orch.risk_manager.cfg.account_equity != starting_equity

    def test_run_ignores_a_bad_equity_reading_and_keeps_the_last_good_value(self, tmp_path, monkeypatch):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        monkeypatch.setattr(orch.broker, "get_equity", lambda mark_prices=None: float("nan"))

        orch.run(max_bars=1)

        # A bad reading (e.g. a transient API hiccup) must not corrupt
        # sizing -- cfg.account_equity should still hold whatever valid
        # value it had before (the original config.yaml default here).
        assert orch.risk_manager.cfg.account_equity > 0
        import math
        assert math.isfinite(orch.risk_manager.cfg.account_equity)


class TestPerformanceHistoryTracking:
    """Regression coverage for task #16 (the multi-ticker backtest
    harness): before equity_curve/trade_log existed, there was no way to
    recover a full-resolution equity path or a per-trade P&L series from
    a finished run -- DriftMonitor's own equity window is intentionally
    bounded and gets overwritten as it rolls, so it can't answer "what
    was the Sharpe ratio / max drawdown over this entire multi-thousand-
    bar backtest?". scripts/backtest_portfolio.py and
    src/analysis/metrics.py both consume these two lists directly."""

    def test_equity_curve_grows_by_exactly_one_entry_per_bar(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        assert orch.equity_curve == []
        orch.run(max_bars=5)
        assert len(orch.equity_curve) == 5
        # Every entry must be a real, finite equity value -- never a
        # placeholder/default -- so metrics computed over it are
        # meaningful.
        assert all(isinstance(e, float) and e > 0 for e in orch.equity_curve)

    def test_trade_log_records_a_filled_trade_with_the_exact_pnl_fed_to_risk_manager(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        assert orch.trade_log == []
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0, entry_price=100.0)

        orch._resolve_matured(ticker, _FixedCloseBar(90.0))

        assert len(orch.trade_log) == 1
        entry = orch.trade_log[0]
        assert entry["ticker"] == ticker
        assert entry["size_pct_equity"] == 10.0
        # Same number this test's sibling class (TestRiskManagerReceivesRealizedPnl)
        # already confirms lands on risk_manager.state.daily_pnl_pct -- the
        # backtest-reported trade log and the risk manager's own
        # bookkeeping must never be able to disagree about this value.
        assert entry["pnl_pct_of_equity"] == pytest.approx(orch.risk_manager.state.daily_pnl_pct)
        assert entry["pnl_pct_of_equity"] == pytest.approx(-1.0)

    def test_trade_log_excludes_unsized_predictions(self, tmp_path):
        """A hold / zero-confidence / risk-capped / rejected-fill
        prediction (size_pct_equity == 0.0) is not a realized trade and
        must not appear in trade_log -- otherwise profit_factor/win_rate
        computed from it would be diluted by trades that never
        happened."""
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        _enqueue_pending(orch, ticker, action=1, size_pct_equity=0.0, entry_price=100.0)

        orch._resolve_matured(ticker, _FixedCloseBar(90.0))

        assert orch.trade_log == []

    def test_trade_log_accumulates_across_many_resolved_trades(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        for i in range(4):
            _enqueue_pending(orch, ticker, action=2, size_pct_equity=5.0, entry_price=100.0)
            close = 110.0 if i % 2 == 0 else 95.0  # alternate win/loss
            orch._resolve_matured(ticker, _FixedCloseBar(close))
        assert len(orch.trade_log) == 4
        pnls = [t["pnl_pct_of_equity"] for t in orch.trade_log]
        assert sum(1 for p in pnls if p > 0) == 2
        assert sum(1 for p in pnls if p < 0) == 2


class TestGrossExposureTracking:
    """Regression coverage for a bug found alongside the account-equity
    one: RiskState.open_notional_pct -- the number size_order()'s
    portfolio-wide gross-exposure cap (cfg.max_gross_exposure_pct) checks
    against -- was never computed or fed to the risk manager anywhere in
    the codebase. _gross_exposure_pct() (reading real broker positions)
    and its wiring into run() via risk_manager.update_open_exposure()
    close that gap -- see RiskManager.update_open_exposure's docstring
    for the full consequence (the cap was silently a no-op, forever)."""

    def test_zero_with_no_open_positions(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        assert orch._gross_exposure_pct({}, equity=100_000.0) == 0.0

    def test_reflects_an_open_positions_notional_marked_to_current_price(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        # 100 shares bought at 50, now marked at 60 -- notional must use
        # the CURRENT mark price (60), not the stale avg_price (50).
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=100.0, avg_price=50.0)
        pct = orch._gross_exposure_pct({ticker: 60.0}, equity=100_000.0)
        assert pct == pytest.approx(100.0 * 60.0 / 100_000.0 * 100.0)

    def test_falls_back_to_avg_price_when_no_mark_price_is_known(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=50.0, avg_price=20.0)
        pct = orch._gross_exposure_pct({}, equity=100_000.0)
        assert pct == pytest.approx(50.0 * 20.0 / 100_000.0 * 100.0)

    def test_short_positions_contribute_their_absolute_notional(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=-40.0, avg_price=25.0)
        pct = orch._gross_exposure_pct({ticker: 25.0}, equity=100_000.0)
        assert pct == pytest.approx(40.0 * 25.0 / 100_000.0 * 100.0)

    def test_zero_or_negative_equity_returns_zero_rather_than_dividing_by_it(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=10.0, avg_price=25.0)
        assert orch._gross_exposure_pct({ticker: 25.0}, equity=0.0) == 0.0
        assert orch._gross_exposure_pct({ticker: 25.0}, equity=-100.0) == 0.0

    def test_run_feeds_real_gross_exposure_into_the_risk_manager(self, tmp_path):
        """The actual end-to-end effect of the fix: before it existed,
        risk_manager.state.open_notional_pct stayed hardcoded at 0.0 for
        an entire run no matter how many positions were open (verified
        directly on the real pipeline: 271 trades, zero exposure
        tracking, in the investigation that found this bug)."""
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=100.0, avg_price=50.0)

        orch.run(max_bars=1)

        assert orch.risk_manager.state.open_notional_pct > 0.0


class TestDailyCounterReset:
    """Regression coverage for a bug found by auditing every public
    RiskManager method for whether it's actually called anywhere in
    production code: reset_daily_counters() was fully implemented and
    unit-tested in isolation, but never invoked by Orchestrator --
    state.daily_pnl_pct (what max_daily_loss_pct's kill-switch check
    compares against) was an all-time cumulative total, never reset per
    calendar day. _maybe_reset_daily_counters() closes that gap by
    detecting a calendar-day boundary crossing in the bar timestamp
    stream -- see its docstring in src/orchestrator.py for the two
    concrete failure modes this caused."""

    def test_the_very_first_bar_ever_seen_does_not_trigger_a_reset(self, tmp_path):
        """The first call just has nothing to compare against yet -- it
        must record today's date, not treat "no prior reset" as if it
        were a day-boundary crossing and wipe out same-day P&L."""
        orch = _make_orchestrator(tmp_path)
        orch.risk_manager.state.daily_pnl_pct = -1.5
        assert orch._last_daily_reset_date is None

        orch._maybe_reset_daily_counters(pd.Timestamp("2024-01-02 09:30"))

        assert orch.risk_manager.state.daily_pnl_pct == -1.5
        assert orch._last_daily_reset_date == pd.Timestamp("2024-01-02").date()

    def test_bars_within_the_same_calendar_day_do_not_reset_the_counter(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch._maybe_reset_daily_counters(pd.Timestamp("2024-01-02 09:30"))
        orch.risk_manager.state.daily_pnl_pct = -2.0

        orch._maybe_reset_daily_counters(pd.Timestamp("2024-01-02 15:45"))

        assert orch.risk_manager.state.daily_pnl_pct == -2.0

    def test_crossing_a_calendar_day_boundary_resets_the_counter_to_zero(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch._maybe_reset_daily_counters(pd.Timestamp("2024-01-02 09:30"))
        orch.risk_manager.state.daily_pnl_pct = -3.7

        orch._maybe_reset_daily_counters(pd.Timestamp("2024-01-03 09:30"))

        assert orch.risk_manager.state.daily_pnl_pct == 0.0
        assert orch._last_daily_reset_date == pd.Timestamp("2024-01-03").date()

    def test_a_reset_persists_risk_state_when_a_state_path_is_configured(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch.risk_state_path = tmp_path / "risk_state.json"
        orch._maybe_reset_daily_counters(pd.Timestamp("2024-01-02 09:30"))
        orch.risk_manager.state.daily_pnl_pct = -1.0

        orch._maybe_reset_daily_counters(pd.Timestamp("2024-01-03 09:30"))

        fresh = RiskManager(RiskConfig())
        assert fresh.load_state(orch.risk_state_path) is True
        assert fresh.state.daily_pnl_pct == 0.0

    def test_run_wires_the_reset_into_the_main_loop(self, tmp_path):
        """A loose end-to-end smoke check that run() actually calls
        _maybe_reset_daily_counters once per bar (not just that the
        helper itself works in isolation) -- a single bar is enough to
        confirm _last_daily_reset_date gets initialized from the real
        feed's bar timestamps."""
        orch = _make_orchestrator(tmp_path)
        assert orch._last_daily_reset_date is None

        orch.run(max_bars=1)

        assert orch._last_daily_reset_date is not None


class TestStopLossEnforcement:
    """Regression coverage for a real bug found by checking whether
    RiskManager.size_order()'s `stop_loss_pct` was actually read anywhere
    downstream: it wasn't. A filled position was only ever closed when
    its prediction matured at `mature_at_count` -- a full
    `label_cfg.horizon` bars later (15 by default) -- however far the
    price moved against it in the meantime. A "per-position stop loss"
    was documented but nothing enforced it, so a position could lose far
    more than its configured stop before ever being closed.
    `_check_stop_losses()` closes this gap by checking every open FILLED
    position's bar-over-bar high/low against its OWN stop level (see
    `_enqueue_pending`'s `stop_loss_pct` param -- these tests all use a
    flat 3.0 so the arithmetic below is easy to follow; TestStopLossVolScaling
    covers the per-ticker volatility scaling itself) every bar, exiting
    early (at the stop price) the moment it's breached."""

    def test_a_long_position_exits_early_when_the_bars_low_breaches_the_stop(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        # 97.0 stop level, below -- the pending entry's own stop_loss_pct
        # (3.0, _enqueue_pending's default), not a global config value.
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0,
                          entry_price=100.0, mature_at_count=10_000)

        # Breaches the 97.0 stop intrabar (low=96) then recovers to close
        # at 98 -- a real stop order would still have filled on the way
        # down, so this must count as stopped out, not survive because
        # the bar's CLOSE never crossed the line.
        orch._check_stop_losses(ticker, _OHLCBar(high=99.0, low=96.0, close=98.0))

        assert len(orch._pending[ticker]) == 0, "stopped position must leave the pending queue"
        assert len(orch.trade_log) == 1
        assert orch.trade_log[0]["pnl_pct_of_equity"] == pytest.approx(10.0 * -0.03)

    def test_a_short_position_exits_early_when_the_bars_high_breaches_the_stop(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        _enqueue_pending(orch, ticker, action=0, size_pct_equity=10.0,
                          entry_price=100.0, mature_at_count=10_000)

        # Short's stop is ABOVE entry (103.0) -- breached intrabar by
        # high=104, recovers to close at 100.
        orch._check_stop_losses(ticker, _OHLCBar(high=104.0, low=99.0, close=100.0))

        assert len(orch._pending[ticker]) == 0
        assert len(orch.trade_log) == 1
        assert orch.trade_log[0]["pnl_pct_of_equity"] == pytest.approx(10.0 * -0.03)

    def test_a_position_within_the_stop_band_survives_and_stays_pending(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0,
                          entry_price=100.0, mature_at_count=10_000)

        # Low of 98 never reaches the 97.0 stop level.
        orch._check_stop_losses(ticker, _OHLCBar(high=101.0, low=98.0, close=99.0))

        assert len(orch._pending[ticker]) == 1
        assert orch.trade_log == []

    def test_an_unfilled_prediction_is_never_stopped_out(self, tmp_path):
        """size_pct_equity == 0.0 means the order was never actually
        filled (hold / low-confidence / risk-capped / rejected) -- there
        is no real position to stop out of, and resolving it early would
        corrupt its training label with a synthetic outcome instead of
        what actually happens over the label's real horizon."""
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=0.0,
                          entry_price=100.0, mature_at_count=10_000)

        # A huge adverse move that would obviously breach any stop, if
        # there were a real position.
        orch._check_stop_losses(ticker, _OHLCBar(high=101.0, low=50.0, close=99.0))

        assert len(orch._pending[ticker]) == 1, "unfilled prediction must stay pending, not be stopped out"
        assert orch.trade_log == []

    def test_stop_loss_is_checked_before_maturity_so_a_breach_resolves_exactly_once(self, tmp_path):
        """A position that would both breach its stop AND naturally
        mature on the same bar must be resolved via the stop (which
        pulls it out of `_pending` first), never both -- run()'s wiring
        calls _check_stop_losses before _resolve_matured for exactly
        this reason."""
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0,
                          entry_price=100.0, mature_at_count=0)  # already "matured" by bar_count

        bar = _OHLCBar(high=99.0, low=90.0, close=95.0)
        orch._check_stop_losses(ticker, bar)
        orch._resolve_matured(ticker, bar)

        assert len(orch.trade_log) == 1, "must resolve exactly once, not once per path"
        # Exits at the STOP price (97.0 -> -3%), not the bar's close
        # (95.0 -> -5%) -- proof _resolve_matured never got a chance to
        # re-process it using the close.
        assert orch.trade_log[0]["pnl_pct_of_equity"] == pytest.approx(10.0 * -0.03)

    def test_run_checks_stop_losses_once_per_bar(self, tmp_path, monkeypatch):
        orch = _make_orchestrator(tmp_path)
        calls = []
        original = orch._check_stop_losses
        def spy(ticker, bar):
            calls.append(ticker)
            return original(ticker, bar)
        monkeypatch.setattr(orch, "_check_stop_losses", spy)

        orch.run(max_bars=4)

        assert calls == [orch.tickers[0]] * 4


class TestPerTickerExposureTracking:
    """Regression coverage for a bug found by extending the dead-code
    audit one step further than RiskManager's own methods, to what
    size_order() actually checks max_position_pct against:
    RiskManager.update_per_ticker_exposure() -- and the data this file
    feeds it, _ticker_notional_pct() -- close the gap documented in that
    method's docstring (max_position_pct was only ever checked against
    each new order in isolation, never against a ticker's
    already-accumulated position)."""

    def test_zero_with_no_open_position_in_that_ticker(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        assert orch._ticker_notional_pct(ticker, {}, equity=100_000.0) == 0.0

    def test_reflects_the_tickers_position_notional_marked_to_current_price(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=100.0, avg_price=50.0)
        pct = orch._ticker_notional_pct(ticker, {ticker: 60.0}, equity=100_000.0)
        assert pct == pytest.approx(100.0 * 60.0 / 100_000.0 * 100.0)

    def test_falls_back_to_avg_price_when_no_mark_price_is_known(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=50.0, avg_price=20.0)
        pct = orch._ticker_notional_pct(ticker, {}, equity=100_000.0)
        assert pct == pytest.approx(50.0 * 20.0 / 100_000.0 * 100.0)

    def test_zero_or_negative_equity_returns_zero_rather_than_dividing_by_it(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=10.0, avg_price=25.0)
        assert orch._ticker_notional_pct(ticker, {ticker: 25.0}, equity=0.0) == 0.0
        assert orch._ticker_notional_pct(ticker, {ticker: 25.0}, equity=-100.0) == 0.0

    def test_run_feeds_real_per_ticker_exposure_into_the_risk_manager(self, tmp_path):
        """The actual end-to-end effect: before this fix,
        risk_manager.state.per_ticker_notional_pct never existed/was
        never populated, no matter how large a position accumulated in
        one ticker (verified directly on the real pipeline: a single
        ticker's position reached 17.9% of equity against a configured
        10% cap, in the investigation that found this bug)."""
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=100.0, avg_price=50.0)

        orch.run(max_bars=1)

        assert orch.risk_manager.state.per_ticker_notional_pct.get(ticker, 0.0) > 0.0


class TestRiskStateLogging:
    """Regression coverage for DecisionLogger.log_risk_state: before the
    exposure-cap fixes above, open_notional_pct/per_ticker_notional_pct/
    daily_pnl_pct were either permanently 0.0 or an uncapped running
    total, so there was nothing meaningful to log here. Now that they're
    real, run() writes one "risk_state" row per bar so a dashboard can
    show the configured caps are actually being respected over time."""

    def test_run_writes_one_risk_state_row_per_bar(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch.run(max_bars=3)

        lines = orch.logger.log_path.read_text().strip().split("\n")
        rows = [json.loads(l) for l in lines]
        risk_state_rows = [r for r in rows if r["type"] == "risk_state"]
        assert len(risk_state_rows) == 3

    def test_logged_row_carries_the_real_state_and_configured_caps(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=100.0, avg_price=50.0)

        orch.run(max_bars=1)

        lines = orch.logger.log_path.read_text().strip().split("\n")
        rows = [json.loads(l) for l in lines]
        row = next(r for r in rows if r["type"] == "risk_state")
        assert row["open_notional_pct"] == pytest.approx(orch.risk_manager.state.open_notional_pct)
        assert row["per_ticker_notional_pct"].get(ticker, 0.0) > 0.0
        assert row["max_gross_exposure_pct"] == orch.risk_manager.cfg.max_gross_exposure_pct
        assert row["max_position_pct"] == orch.risk_manager.cfg.max_position_pct
        assert row["max_daily_loss_pct"] == orch.risk_manager.cfg.max_daily_loss_pct


class TestRealBrokerCloseOnResolution:
    """Regression coverage for a gap found by a dead-code audit of this
    exact file, after the stop-loss/per-ticker-cap fixes above: both exit
    paths in _resolve_one (natural maturity via _resolve_matured, and the
    early hard stop-loss via _check_stop_losses) updated every piece of
    SYNTHETIC bookkeeping -- drift monitor, replay buffer, trade_log, risk
    manager daily P&L/consecutive-losses -- but never submitted a real
    closing order to the broker. Confirmed directly by grep before any
    fix: `submit_order` had exactly one call site in this file (run()'s
    new-signal path), so broker.positions/.cash were never touched by an
    exit, no matter how it happened. The practical consequence: the real
    (paper or live) position just kept sitting open past its "stop" or
    its label horizon, continuing to drift with the market, until some
    unrelated future signal happened to net it down -- which could be
    never. Broker.close_quantity + this wiring close that gap."""

    def test_maturity_resolution_actually_closes_the_real_broker_position(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        # Stand in for a real prior fill: 100 shares long at 100.0.
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=100.0, avg_price=100.0)
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0,
                          entry_price=100.0, mature_at_count=0, filled_quantity=100.0)

        orch._resolve_matured(ticker, _FixedCloseBar(110.0))

        assert orch.broker.get_position(ticker).quantity == pytest.approx(0.0, abs=1e-6), (
            "maturity resolution must actually flatten the real broker position "
            "it represents, not just update synthetic bookkeeping"
        )

    def test_stop_loss_resolution_actually_closes_the_real_broker_position(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=50.0, avg_price=100.0)
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0,
                          entry_price=100.0, mature_at_count=10_000, filled_quantity=50.0)

        orch._check_stop_losses(ticker, _OHLCBar(high=99.0, low=96.0, close=98.0))

        assert orch.broker.get_position(ticker).quantity == pytest.approx(0.0, abs=1e-6)

    def test_closing_a_short_position_buys_back_the_exact_quantity(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=-30.0, avg_price=100.0)
        _enqueue_pending(orch, ticker, action=0, size_pct_equity=10.0,
                          entry_price=100.0, mature_at_count=0, filled_quantity=30.0)

        orch._resolve_matured(ticker, _FixedCloseBar(90.0))

        assert orch.broker.get_position(ticker).quantity == pytest.approx(0.0, abs=1e-6)

    def test_one_entrys_close_does_not_disturb_a_second_still_open_entry_in_the_same_ticker(self, tmp_path):
        """Two overlapping predictions on the same ticker (the FIFO
        _pending queue can hold more than one at once, e.g. a new signal
        every bar against a 15-bar label horizon) must each net out only
        their own contribution when they resolve -- resolving the first
        must not touch the second's still-open share of the position."""
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=150.0, avg_price=100.0)
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0,
                          entry_price=100.0, mature_at_count=0, filled_quantity=100.0)
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=5.0,
                          entry_price=100.0, mature_at_count=10_000, filled_quantity=50.0)

        orch._resolve_matured(ticker, _FixedCloseBar(110.0))

        assert len(orch._pending[ticker]) == 1, "only the matured entry should resolve"
        assert orch.broker.get_position(ticker).quantity == pytest.approx(50.0), (
            "closing the first entry's 100 shares must leave exactly the second "
            "entry's still-open 50 shares untouched"
        )

    def test_unfilled_prediction_never_submits_a_closing_order(self, tmp_path, monkeypatch):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        calls = []
        monkeypatch.setattr(orch.broker, "close_quantity",
                             lambda *a, **k: calls.append((a, k)))
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=0.0,
                          entry_price=100.0, mature_at_count=0)

        orch._resolve_matured(ticker, _FixedCloseBar(90.0))

        assert calls == [], "no real fill happened -- there is nothing to close"

    def test_close_order_is_logged_when_it_fills(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        orch.broker.positions[ticker] = Position(ticker=ticker, quantity=100.0, avg_price=100.0)
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0,
                          entry_price=100.0, mature_at_count=0, filled_quantity=100.0)

        orch._resolve_matured(ticker, _FixedCloseBar(110.0))

        lines = orch.logger.log_path.read_text().strip().split("\n")
        rows = [json.loads(l) for l in lines]
        close_rows = [r for r in rows if r["type"] == "close_order"]
        assert len(close_rows) == 1
        assert close_rows[0]["filled"] is True
        assert close_rows[0]["requested_quantity"] == pytest.approx(100.0)
        assert close_rows[0]["filled_quantity"] == pytest.approx(100.0)

    def test_a_close_that_fails_to_fill_is_logged_as_unfilled_not_silently_dropped(self, tmp_path, monkeypatch):
        """E.g. an AlpacaBroker order that doesn't confirm within the poll
        window: close_quantity returns None. The synthetic bookkeeping
        (training label, trade_log, risk counters) still proceeds --
        the prediction was still right or wrong regardless of whether the
        real order executed -- but this must be visible in the decision
        log instead of indistinguishable from a real close that worked."""
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        monkeypatch.setattr(orch.broker, "close_quantity", lambda *a, **k: None)
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0,
                          entry_price=100.0, mature_at_count=0, filled_quantity=100.0)

        orch._resolve_matured(ticker, _FixedCloseBar(110.0))

        lines = orch.logger.log_path.read_text().strip().split("\n")
        rows = [json.loads(l) for l in lines]
        close_rows = [r for r in rows if r["type"] == "close_order"]
        assert len(close_rows) == 1
        assert close_rows[0]["filled"] is False
        assert close_rows[0]["requested_quantity"] == pytest.approx(100.0)
        # The trade still gets graded and logged even though the real
        # close didn't confirm -- that's the existing, intentional
        # separation between "was the prediction right" and "did the
        # real order execute."
        assert len(orch.trade_log) == 1

    def test_run_end_to_end_a_filled_position_is_flattened_at_maturity(self, tmp_path, monkeypatch):
        """Full loop, not a reimplementation: force a real fill on the
        first bar via run(), fast-forward past its label horizon by
        hand, and confirm the real broker position this fill opened is
        actually flat afterward -- the end-to-end version of the
        maturity-close test above, going through the real submit_order
        path instead of a hand-seeded Position."""
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]

        def make_sizing(signal, vol):
            # Force a fill regardless of the (untrained, effectively
            # random) model's actual confidence, but -- same as the real
            # size_order() always does -- keep the SAME direction the
            # model actually predicted (signal.action). Forcing a
            # different direction here than pred["action"] would create
            # a mismatch size_order() itself can never produce (it never
            # flips a signal's direction, only sizes it to zero), which
            # would make this test exercise a scenario the real pipeline
            # can't reach.
            return {"action": signal.action, "size_pct_equity": 10.0, "stop_loss_pct": 3.0,
                    "reason": "sized", "ticker": signal.ticker, "model_version": signal.model_version}

        monkeypatch.setattr(orch.risk_manager, "size_order", make_sizing)
        orch.run(max_bars=1)

        assert len(orch._pending[ticker]) == 1
        pending = orch._pending[ticker][0]
        assert pending["size_pct_equity"] > 0, "the order must have actually filled"
        assert orch.broker.get_position(ticker).quantity != 0.0, "the real fill must be reflected in the broker"

        # Force maturity right now and resolve it directly (bypassing the
        # label horizon's 15-bar wait, same pattern _enqueue_pending's
        # mature_at_count=0 tests above use).
        pending["mature_at_count"] = 0
        orch._resolve_matured(ticker, _FixedCloseBar(orch._history[ticker]["close"].iloc[-1] * 1.02))

        assert orch.broker.get_position(ticker).quantity == pytest.approx(0.0, abs=1e-6), (
            "the position run() actually opened must be actually closed at maturity, "
            "not left open while only the synthetic bookkeeping updates"
        )


class TestStopLossVolScaling:
    """Regression coverage for the Orchestrator-side wiring of
    volatility-scaled stops (see TestStopLossVolScaling in test_risk.py
    for the RiskManager-side computation itself): each pending entry
    carries its OWN stop_loss_pct, computed from that ticker's realized
    vol at the moment it was sized, and _check_stop_losses must read
    that per-entry value, not one value shared by the whole ticker or
    the whole run."""

    def test_two_entries_with_different_stops_breach_independently(self, tmp_path):
        """Same adverse move (low=96.5), two otherwise-identical
        positions -- one with a tight (low-vol) stop that it breaches,
        one with a wide (high-vol) stop that it doesn't -- on two
        different (synthetic, not in orch.tickers) ticker keys of the
        same orchestrator's _pending dict. Proves _check_stop_losses
        reads each entry's own stop_loss_pct rather than one value
        shared across every position."""
        orch = _make_orchestrator(tmp_path)
        tight, wide = orch.tickers[0], "SYNTHETIC_WIDE"
        orch._pending[wide] = deque()
        orch._bar_count[wide] = 0

        _enqueue_pending(orch, tight, action=2, size_pct_equity=10.0, entry_price=100.0,
                          mature_at_count=10_000, stop_loss_pct=2.0)   # stop at 98.0
        _enqueue_pending(orch, wide, action=2, size_pct_equity=10.0, entry_price=100.0,
                          mature_at_count=10_000, stop_loss_pct=6.0)   # stop at 94.0

        bar = _OHLCBar(high=101.0, low=96.5, close=99.0)
        orch._check_stop_losses(tight, bar)
        orch._check_stop_losses(wide, bar)

        assert len(orch._pending[tight]) == 0, "the tight 2% stop (98.0) must have breached at low=96.5"
        assert len(orch._pending[wide]) == 1, "the wide 6% stop (94.0) must NOT have breached at low=96.5"

    def test_the_stop_level_used_is_the_entrys_own_stop_loss_pct(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=10.0, entry_price=100.0,
                          mature_at_count=10_000, stop_loss_pct=5.0)  # stop at 95.0

        # Breaches a flat 3% stop (97.0) but NOT this entry's real 5% stop (95.0).
        orch._check_stop_losses(ticker, _OHLCBar(high=99.0, low=96.0, close=98.0))
        assert len(orch._pending[ticker]) == 1, "must survive -- 96.0 is above this entry's own 95.0 stop"

        # Now actually breaches the 5% stop.
        orch._check_stop_losses(ticker, _OHLCBar(high=96.0, low=94.5, close=95.5))
        assert len(orch._pending[ticker]) == 0
        assert orch.trade_log[0]["pnl_pct_of_equity"] == pytest.approx(10.0 * -0.05)

    def test_run_stores_the_real_risk_managers_vol_scaled_stop_on_the_pending_entry(self, tmp_path, monkeypatch):
        """End-to-end: run() must store size_order()'s real, computed
        stop_loss_pct on the pending entry -- not a hardcoded value, and
        not silently dropped."""
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        captured_vol = {}

        real_stop_loss_pct_for = orch.risk_manager.stop_loss_pct_for

        def spy_stop_loss_pct_for(vol):
            captured_vol["vol"] = vol
            return real_stop_loss_pct_for(vol)

        monkeypatch.setattr(orch.risk_manager, "stop_loss_pct_for", spy_stop_loss_pct_for)

        def make_sizing(signal, vol):
            return {"action": signal.action, "size_pct_equity": 10.0,
                    "stop_loss_pct": orch.risk_manager.stop_loss_pct_for(vol),
                    "reason": "sized", "ticker": signal.ticker, "model_version": signal.model_version}

        monkeypatch.setattr(orch.risk_manager, "size_order", make_sizing)
        orch.run(max_bars=1)

        assert len(orch._pending[ticker]) == 1
        pending = orch._pending[ticker][0]
        assert "vol" in captured_vol, "the real realized_vol computed this bar must have driven the stop"
        assert pending["stop_loss_pct"] == pytest.approx(real_stop_loss_pct_for(captured_vol["vol"]))

    def test_an_unfilled_predictions_stop_loss_pct_is_none_and_is_never_read(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        ticker = orch.tickers[0]
        _enqueue_pending(orch, ticker, action=2, size_pct_equity=0.0,
                          entry_price=100.0, mature_at_count=10_000, stop_loss_pct=None)

        # Would obviously breach any real stop -- must survive untouched,
        # since size_pct_equity <= 0 means there's no real position and
        # _check_stop_losses must never try to read the None stop_loss_pct.
        orch._check_stop_losses(ticker, _OHLCBar(high=101.0, low=1.0, close=99.0))

        assert len(orch._pending[ticker]) == 1
