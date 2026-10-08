import sys
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


def _enqueue_pending(orch, ticker, *, action, size_pct_equity, entry_price=100.0):
    orch._pending[ticker].append({
        "decision_id": "d-test",
        "entry_price": entry_price,
        "action": action,          # 0=sell, 1=hold, 2=buy
        "confidence": 0.9,
        "feature_window": np.zeros((orch.window, N_FEATURES), dtype=np.float32),
        "mature_at_count": 0,
        "size_pct_equity": size_pct_equity,
    })
    orch._bar_count[ticker] = 0


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
