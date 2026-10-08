"""
The main loop: ties data ingestion, inference, risk, execution, logging,
and continual training together.

One iteration, per incoming bar:

    1.  Append the bar to that ticker's rolling history.
    2.  Recompute features over the (bounded) rolling history and, if
        there's enough warm-up data, build the latest feature window and
        run inference -> a Signal.
    3.  Risk-size the signal (or get back "no trade").
    4.  Submit an order to the broker if sized > 0; log every step.
    5.  Check every *earlier* FILLED position still open against its hard
        stop-loss (this bar's high/low vs. the stop level); close out any
        breach early, at the stop price.
    6.  Check whether any *earlier* predictions have now matured (their
        label horizon has elapsed, and didn't already exit via a stop
        above): if so, compute the realized outcome, feed it to the
        drift monitor and the replay buffer (this is the "continuous
        supervision" loop), and log it.
    7.  Update the equity curve, check the kill switch, and periodically
        let the continual trainer attempt a (gated) retrain.

This file intentionally has no network/broker-specific code in it -- swap
`SyntheticFeed`/`YFinanceFeed` for a real feed and `PaperBroker` for a real
broker, and this loop does not change.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch

from src.data.buffer import ReplayBuffer
from src.data.feed import MarketDataFeed
from src.data.features import LabelConfig, build_windows, compute_features
from src.execution.broker import Broker
from src.model.network import TradingNet
from src.model.signals import to_signal
from src.monitor.logger import DecisionLogger
from src.risk.manager import RiskManager
from src.training.drift import DriftMonitor
from src.training.trainer import ContinualTrainer


class Orchestrator:
    def __init__(self, tickers, feed: MarketDataFeed, model: TradingNet,
                 trainer: ContinualTrainer, risk_manager: RiskManager,
                 broker: Broker, logger: DecisionLogger, drift_monitor: DriftMonitor,
                 window: int = 60, label_cfg: LabelConfig | None = None,
                 max_history: int = 400, device: str = "cpu", warmup_bars: int = 120,
                 risk_state_path: str | None = None, drift_state_path: str | None = None):
        self.tickers = list(tickers)
        self.feed = feed
        self.trainer = trainer
        self.risk_manager = risk_manager
        self.broker = broker
        self.logger = logger
        self.drift_monitor = drift_monitor
        self.window = window
        self.label_cfg = label_cfg or LabelConfig()
        self.max_history = max_history
        self.device = device
        self.warmup_bars = warmup_bars
        # When set, risk_manager/drift_monitor state is restored from these
        # paths at the start of run() and re-saved after every event that
        # changes them (not just on clean shutdown -- see save_state()'s
        # docstrings on RiskManager/DriftMonitor for why). None (the
        # default) means "don't persist" -- the original in-memory-only
        # behavior, e.g. for backtests/tests that shouldn't touch disk.
        self.risk_state_path = risk_state_path
        self.drift_state_path = drift_state_path

        self.buffer = ReplayBuffer(window=window, n_features=model.cfg.n_features)
        self._history: dict[str, pd.DataFrame] = {t: pd.DataFrame(columns=["open", "high", "low", "close", "volume"]) for t in self.tickers}
        self._bar_count: dict[str, int] = defaultdict(int)
        self._pending: dict[str, deque] = {t: deque() for t in self.tickers}  # per-ticker FIFO of open predictions
        self.retrain_events: list[dict] = []
        # Full-resolution, unbounded history for offline analysis --
        # DriftMonitor keeps its own equity window too, but that one is
        # intentionally bounded (cfg.window * 5) for live monitoring, so
        # it can't answer "what was the Sharpe/drawdown over this whole
        # multi-thousand-bar backtest?". These two lists are what
        # scripts/backtest_portfolio.py and src/analysis/metrics.py
        # consume; see _resolve_matured() for what makes it into
        # trade_log and why (only trades that actually filled).
        self.equity_curve: list[float] = []
        self.trade_log: list[dict] = []
        # Tracks the calendar date (bar.timestamp's .date()) of the last
        # risk_manager.reset_daily_counters() call, so run() can detect a
        # new trading day starting and reset state.daily_pnl_pct for it.
        # None until the first bar is seen.
        self._last_daily_reset_date = None

    # -- internal helpers -------------------------------------------------

    def _update_history(self, bar) -> None:
        row = pd.DataFrame(
            [[bar.open, bar.high, bar.low, bar.close, bar.volume]],
            columns=["open", "high", "low", "close", "volume"], index=[bar.timestamp],
        )
        hist = pd.concat([self._history[bar.ticker], row])
        if len(hist) > self.max_history:
            hist = hist.iloc[-self.max_history:]
        self._history[bar.ticker] = hist

    def _try_predict(self, ticker: str, timestamp) -> dict | None:
        hist = self._history[ticker]
        if len(hist) < self.warmup_bars:
            return None
        feats = compute_features(hist)
        X, idxs = build_windows(feats, None, self.window)
        if len(X) == 0:
            return None
        x = torch.from_numpy(X[-1:]).to(self.device)
        # Always go through the trainer's current model, never a separately
        # held reference: ContinualTrainer.maybe_retrain() promotes a
        # challenger by rebinding its OWN self.model to a new object
        # (self.model = challenger), not by mutating the existing model in
        # place. A reference captured once (e.g. self.model set in
        # __init__) would silently keep pointing at the stale pre-promotion
        # model forever after the first promoted retrain, even while the
        # logs correctly report a newer model_version.
        out = self.trainer.model.predict(x)
        realized_vol = float(feats["realized_vol_15"].iloc[-1]) if not pd.isna(feats["realized_vol_15"].iloc[-1]) else 0.01
        return {
            "action": int(out["action"][0].item()),
            "confidence": float(out["confidence"][0].item()),
            "expected_return": float(out["expected_return"][0].item()),
            "feature_window": X[-1],
            "realized_vol": realized_vol,
        }

    def _resolve_one(self, ticker: str, p: dict, exit_price: float, timestamp) -> None:
        """Shared bookkeeping for closing out one pending prediction,
        however it closed: naturally at its label horizon
        (`_resolve_matured`, `exit_price` = that bar's close) or early,
        via the hard stop-loss (`_check_stop_losses`, `exit_price` = the
        stop level itself). Both paths must update the drift monitor,
        replay buffer, risk manager, and trade_log identically -- this
        used to be duplicated inline in `_resolve_matured` alone; factored
        out so the two exit paths can never quietly drift out of sync
        with each other (exactly the failure mode this whole file's
        safety fixes keep guarding against elsewhere)."""
        realized_ret = (exit_price - p["entry_price"]) / p["entry_price"]
        deadband = self.label_cfg.deadband_bps / 10_000.0
        true_action = 1 if realized_ret > deadband else (-1 if realized_ret < -deadband else 0)
        pred_action_signed = p["action"] - 1  # {0,1,2} -> {-1,0,1}
        correct = (pred_action_signed == true_action)

        self.drift_monitor.record_prediction_outcome(correct, p["confidence"])
        if self.drift_state_path is not None:
            self.drift_monitor.save_state(self.drift_state_path)
        self.logger.log_outcome(p["decision_id"], realized_ret, correct)
        self.buffer.add(p["feature_window"], true_action + 1, realized_ret, ticker, timestamp)
        self.trainer.notify_new_samples(1)

        # Feed the REALIZED P&L of actual filled trades into the risk
        # manager's daily-loss / consecutive-loss counters. Before this,
        # nothing in the main loop ever called
        # RiskManager.update_after_trade_result() -- it was exercised
        # only by unit tests -- so cfg.max_daily_loss_pct and
        # cfg.max_consecutive_losses could never trip during real
        # trading, no matter how badly a session went. Only
        # DriftMonitor's hit-rate/brier/equity-drawdown halt (below,
        # after this loop) was ever live. `size_pct_equity` is 0.0 for
        # any prediction that wasn't actually sized/filled (hold, low
        # confidence, risk caps, order rejection -- see `run()`), so
        # those correctly don't count as a realized trade here.
        if p["size_pct_equity"] > 0:
            pnl_pct_of_equity = p["size_pct_equity"] * pred_action_signed * realized_ret
            self.risk_manager.update_after_trade_result(pnl_pct_of_equity)
            if self.risk_state_path is not None:
                self.risk_manager.save_state(self.risk_state_path)
            # Record every REALIZED, actually-filled trade for offline
            # performance analysis (profit factor / win rate / turnover
            # -- see src/analysis/metrics.py). Deliberately the exact
            # same filter (size_pct_equity > 0) and the exact same
            # pnl_pct_of_equity value just fed into the risk manager
            # above, so backtest-reported metrics and the risk
            # manager's own daily P&L tracking can never disagree
            # about which trades counted or by how much.
            self.trade_log.append({
                "ticker": ticker,
                "timestamp": timestamp,
                "pnl_pct_of_equity": pnl_pct_of_equity,
                "size_pct_equity": p["size_pct_equity"],
            })

    def _resolve_matured(self, ticker: str, current_bar) -> None:
        pending = self._pending[ticker]
        while pending and pending[0]["mature_at_count"] <= self._bar_count[ticker]:
            p = pending.popleft()
            self._resolve_one(ticker, p, current_bar.close, current_bar.timestamp)

    def _check_stop_losses(self, ticker: str, current_bar) -> None:
        """Close out, THIS bar, any pending (not-yet-matured) FILLED
        position whose intrabar high/low has breached
        `cfg.hard_stop_loss_pct`.

        Before this, `RiskManager.size_order()` computed and returned a
        `stop_loss_pct` on every sized order, but nothing in
        `Orchestrator` ever read it (confirmed by grep): a filled
        position was only ever closed at `mature_at_count`, i.e. after a
        full `label_cfg.horizon` bars (15, by default) had elapsed,
        however far the price moved against it in the meantime. A
        position sized under the assumption of a "hard stop loss" could
        therefore lose far more than that configured percentage before
        it was ever closed -- the exact same dead-code-safety-check
        pattern as `reset_daily_counters`/`open_notional_pct` above, just
        for the per-position (not daily or portfolio-wide) risk limit.

        Checked against `current_bar.low` (for a long) / `current_bar.high`
        (for a short) rather than `.close`, so a stop that was breached
        and recovered within the same bar is still caught -- a real
        broker's stop order would have filled intrabar too, it wouldn't
        wait to see where the bar closed. The exit is booked at the exact
        stop level, not the bar's actual low/high, which is the standard
        (and conservative-to-model, since real slippage past the stop is
        not modeled either way) simplification for a backtest that only
        has OHLC bars, not a full intrabar price path.

        Deliberately skips any pending entry with `size_pct_equity <= 0`
        (hold / low-confidence / risk-capped / rejected-by-broker -- see
        `run()`): there is no real position to stop out of, and
        resolving it early would corrupt its training label by feeding
        the replay buffer a synthetic early "outcome" instead of what
        actually happened over the full horizon the label is defined
        over. Only a real, filled position can hit a stop."""
        pending = self._pending[ticker]
        if not pending:
            return
        stop_frac = self.risk_manager.cfg.hard_stop_loss_pct / 100.0
        survivors: deque = deque()
        for p in pending:
            if p["size_pct_equity"] <= 0:
                survivors.append(p)
                continue
            direction = p["action"] - 1  # {0,1,2} -> {-1,0,1}; filled => never 0
            exit_price = None
            if direction > 0:
                stop_level = p["entry_price"] * (1 - stop_frac)
                if current_bar.low <= stop_level:
                    exit_price = stop_level
            elif direction < 0:
                stop_level = p["entry_price"] * (1 + stop_frac)
                if current_bar.high >= stop_level:
                    exit_price = stop_level
            if exit_price is not None:
                self._resolve_one(ticker, p, exit_price, current_bar.timestamp)
            else:
                survivors.append(p)
        self._pending[ticker] = survivors

    def _gross_exposure_pct(self, mark_prices: dict[str, float], equity: float) -> float:
        """Aggregate open position notional across every ticker, marked
        to the latest known price, as a percent of current equity --
        exactly the number `risk_manager.update_open_exposure()` needs
        (see that method's docstring for the bug this closes: this value
        was never computed anywhere before, so the portfolio-wide
        gross-exposure cap was always comparing against a hardcoded 0.0).
        Uses `broker.get_position(t)` for every ticker -- cheap (an
        in-memory dict lookup) for `PaperBroker`; for `AlpacaBroker` this
        is one extra API call per ticker per bar, on top of the
        `get_equity()` call already made every bar (see the README)."""
        if equity <= 0:
            return 0.0
        total_notional = 0.0
        for t in self.tickers:
            pos = self.broker.get_position(t)
            if pos.quantity == 0:
                continue
            price = mark_prices.get(t, pos.avg_price)
            total_notional += abs(pos.quantity) * price
        return total_notional / equity * 100.0

    def _ticker_notional_pct(self, ticker: str, mark_prices: dict[str, float], equity: float) -> float:
        """One ticker's own open position notional, marked to the latest
        known price, as a percent of current equity -- exactly the
        number `risk_manager.update_per_ticker_exposure()` needs (see
        that method's docstring for the bug this closes: `max_position_pct`
        was only ever checked against each new order in isolation, never
        against a ticker's already-accumulated exposure). Deliberately a
        separate, smaller computation from `_gross_exposure_pct` above
        rather than a refactor of it -- that method is already covered by
        its own tests and this one only ever needs a single ticker at a
        time (the one about to be sized this bar), not every ticker."""
        if equity <= 0:
            return 0.0
        pos = self.broker.get_position(ticker)
        if pos.quantity == 0:
            return 0.0
        price = mark_prices.get(ticker, pos.avg_price)
        return abs(pos.quantity) * price / equity * 100.0

    def _maybe_reset_daily_counters(self, timestamp) -> None:
        """Calls `risk_manager.reset_daily_counters()` once per calendar
        day, the first time a bar's timestamp's date differs from the
        date of the last reset (or on the very first bar ever seen).

        Before this, `reset_daily_counters()` was implemented but never
        called anywhere in the codebase (confirmed by grep -- not even a
        test exercised it), so `state.daily_pnl_pct` -- what
        `cfg.max_daily_loss_pct`'s kill-switch check compares against --
        was actually a running ALL-TIME total, never reset per trading
        day. Two concrete failure modes that bug: a genuinely bad single
        day could fail to trip the "daily" loss kill switch at all if
        prior days were net positive (the cumulative total stays above
        -max_daily_loss_pct even though today alone breached it); or,
        conversely, a few bad days ago could keep the kill switch
        permanently untrippable or trip it on an otherwise fine day once
        the cumulative total creeps past the threshold -- either way,
        `max_daily_loss_pct` stops meaning what its name says. This
        state is also persisted/resumed across restarts
        (`risk_state_path`), so the corruption would survive a restart
        and keep compounding across calendar days in the saved file."""
        today = pd.Timestamp(timestamp).date()
        if self._last_daily_reset_date is not None and today != self._last_daily_reset_date:
            self.risk_manager.reset_daily_counters()
            if self.risk_state_path is not None:
                self.risk_manager.save_state(self.risk_state_path)
        self._last_daily_reset_date = today

    def _prime_history(self) -> None:
        """Warm up each ticker's rolling history from feed.get_history()
        before the main loop starts, so prediction can begin almost
        immediately instead of waiting out warmup_bars worth of live bars
        (which, on real market data, means waiting warmup_bars minutes --
        two hours at the default of 120 -- before the model says anything).
        Safe to call on any feed: SyntheticFeed/YFinanceFeed advance their
        internal cursor on this call so stream() picks up right after the
        warm-up window instead of replaying it."""
        for t in self.tickers:
            try:
                hist = self.feed.get_history(t, self.max_history)
            except Exception:
                hist = None
            if hist is None or len(hist) == 0:
                continue
            self._history[t] = hist.iloc[-self.max_history:]
            self._bar_count[t] = len(self._history[t])

    # -- main loop ----------------------------------------------------------

    def run(self, max_bars: int | None = None, stop_check=None) -> None:
        """Run the loop. `stop_check`, if given, is a zero-arg callable
        polled once per bar; returning True ends the run cleanly (used for
        e.g. "stop when the market closes" in live trading, without the
        feed itself needing to know about market hours)."""
        self._prime_history()
        n_seen = 0
        mark_prices: dict[str, float] = {}

        for bar in self.feed.stream(self.tickers):
            self._update_history(bar)
            self._bar_count[bar.ticker] += 1
            mark_prices[bar.ticker] = bar.close
            self.logger.log_bar(bar.ticker, bar.timestamp, bar.close)

            # Stop-loss check runs BEFORE maturity resolution: a position
            # whose stop was breached intrabar exits at the stop level,
            # not whatever this bar happens to close at, and is removed
            # from `_pending` so `_resolve_matured` below never
            # double-resolves it.
            self._check_stop_losses(bar.ticker, bar)
            self._resolve_matured(bar.ticker, bar)
            self._maybe_reset_daily_counters(bar.timestamp)

            # Sync position sizing to the broker's actual current equity
            # BEFORE this bar's sizing decision uses it, not after. Before
            # this, risk_manager.cfg.account_equity stayed frozen at
            # whatever static value config.yaml had at startup forever --
            # for AlpacaBroker that's a real paper-account balance that's
            # read fresh every bar anyway (see AlpacaBroker.get_equity()),
            # it just never made it back into the risk manager, so sizing
            # silently drifted from the real account as it compounded
            # gains/losses. Also used below for the drift monitor/log, so
            # this replaces (not duplicates) that later equity fetch.
            equity = self.broker.get_equity(mark_prices)
            self.risk_manager.update_account_equity(equity)
            self.risk_manager.update_open_exposure(self._gross_exposure_pct(mark_prices, equity))
            # Only this bar's own ticker needs a fresh reading here -- it's
            # the only one about to be sized below. Every ticker gets kept
            # current exactly when it matters, each on its own bar.
            self.risk_manager.update_per_ticker_exposure(
                bar.ticker, self._ticker_notional_pct(bar.ticker, mark_prices, equity)
            )
            # One row per bar of exposure/P&L state alongside the caps
            # it's checked against -- see DecisionLogger.log_risk_state's
            # docstring for why this is only worth logging now that these
            # numbers are actually meaningful.
            self.logger.log_risk_state(
                bar.timestamp, self.risk_manager.state.open_notional_pct,
                self.risk_manager.state.per_ticker_notional_pct, self.risk_manager.state.daily_pnl_pct,
                self.risk_manager.cfg.max_gross_exposure_pct, self.risk_manager.cfg.max_position_pct,
                self.risk_manager.cfg.max_daily_loss_pct,
            )

            pred = self._try_predict(bar.ticker, bar.timestamp)
            if pred is not None:
                decision_id = self.logger.log_prediction(
                    bar.ticker, bar.timestamp, self.trainer.model_version(),
                    pred["action"], pred["confidence"], pred["expected_return"],
                )
                signal = to_signal(bar.ticker, bar.timestamp, pred["action"], pred["confidence"],
                                    pred["expected_return"], self.trainer.model_version())
                sizing = self.risk_manager.size_order(signal, pred["realized_vol"])
                self.logger.log_risk_decision(decision_id, {
                    "size_pct_equity": sizing["size_pct_equity"],
                    "reason": sizing["reason"],
                })

                fill = None
                if sizing["size_pct_equity"] > 0:
                    notional = self.risk_manager.cfg.account_equity * sizing["size_pct_equity"] / 100.0
                    fill = self.broker.submit_order(bar.ticker, sizing["action"], notional, bar.close, bar.timestamp)
                    if fill is not None:
                        self.logger.log_fill(decision_id, fill)

                self._pending[bar.ticker].append({
                    "decision_id": decision_id,
                    "entry_price": bar.close,
                    "action": pred["action"],
                    "confidence": pred["confidence"],
                    "feature_window": pred["feature_window"],
                    "mature_at_count": self._bar_count[bar.ticker] + self.label_cfg.horizon,
                    # 0.0 unless a real order was actually filled -- see
                    # _resolve_matured()'s use of this to feed realized P&L
                    # into the risk manager. A rejected/skipped order (fill
                    # is None) must not be counted as a realized trade.
                    "size_pct_equity": sizing["size_pct_equity"] if fill is not None else 0.0,
                })

            self.drift_monitor.record_equity(equity)
            if self.drift_state_path is not None:
                self.drift_monitor.save_state(self.drift_state_path)
            self.logger.log_equity(bar.timestamp, equity)
            self.equity_curve.append(equity)
            halted, reasons = self.drift_monitor.should_halt()
            if halted and not self.risk_manager.kill_switch_engaged():
                self.risk_manager.trip_kill_switch("; ".join(reasons))
                self.logger.log_halt(reasons)
                if self.risk_state_path is not None:
                    self.risk_manager.save_state(self.risk_state_path)

            if self.trainer.ready_to_retrain(self.buffer):
                record = self.trainer.maybe_retrain(self.buffer)
                if record is not None:
                    self.logger.log_retrain(record)
                    self.retrain_events.append(record)

            n_seen += 1
            if max_bars is not None and n_seen >= max_bars:
                break
            if stop_check is not None and stop_check():
                break
