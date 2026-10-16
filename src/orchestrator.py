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
        breach early, at the stop price -- submitting a real closing
        order to the broker for the exact quantity that entry filled,
        not just updating bookkeeping (see Broker.close_quantity).
    6.  Check whether any *earlier* predictions have now matured (their
        label horizon has elapsed, and didn't already exit via a stop
        above): if so, compute the realized outcome, feed it to the
        drift monitor and the replay buffer (this is the "continuous
        supervision" loop), log it, and -- same as step 5 -- submit the
        real closing order that flattens the position this prediction
        opened.
    7.  Update the equity curve, check the kill switch, and periodically
        let the continual trainer attempt a (gated) retrain.

This file intentionally has no network/broker-specific code in it -- swap
`SyntheticFeed`/`YFinanceFeed` for a real feed and `PaperBroker` for a real
broker, and this loop does not change.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.buffer import ReplayBuffer
from src.data.feed import MarketDataFeed
from src.data.features import LabelConfig, build_windows, compute_features
from src.execution.broker import Broker, Position
from src.model.network import TradingNet
from src.model.signals import Action, to_signal
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
                 risk_state_path: str | None = None, drift_state_path: str | None = None,
                 signal_override=None, positions_state_path: str | None = None,
                 protective_stops: bool = True, adopted_horizon_bars: int | None = None):
        self.tickers = list(tickers)
        # Backtest-only: a callable (ticker, timestamp) -> {"action",
        # "confidence", "expected_return"} that replaces the model's
        # entry call (e.g. src/model/dip_buy.DipBuySignals). The model
        # still trains on every outcome exactly as before. None (the
        # default, and always in the live scripts) leaves entries unchanged.
        self.signal_override = signal_override
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
        # Open FILLED positions are persisted here (see _save_positions /
        # _reconcile_positions) so a restart never orphans them. None
        # (backtests, tests) keeps the old in-memory-only behavior.
        self.positions_state_path = positions_state_path
        # Rest a real stop order at the broker for every filled entry, when
        # the broker supports it (AlpacaBroker); a no-op otherwise.
        self.protective_stops = protective_stops
        # Bars a position found at the broker with no saved record gets
        # before its time exit; defaults to the label horizon.
        self.adopted_horizon_bars = adopted_horizon_bars

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
        decision = {
            "action": int(out["action"][0].item()),
            "confidence": float(out["confidence"][0].item()),
            "expected_return": float(out["expected_return"][0].item()),
        }
        # The model's own call is always what gets scored (drift monitor
        # hit rate / Brier, decision log): grading an override's mostly-HOLD
        # calls would trip the kill switch on a score that says nothing
        # about the model or the override.
        model_decision = dict(decision)
        if self.signal_override is not None:
            decision = self.signal_override(ticker, timestamp)
        return {
            **decision,
            "model_action": model_decision["action"],
            "model_confidence": model_decision["confidence"],
            "model_expected_return": model_decision["expected_return"],
            "feature_window": X[-1],
            "realized_vol": realized_vol,
        }

    def _resolve_one(self, ticker: str, p: dict, exit_price: float, timestamp,
                     stop_result: tuple | None = None) -> None:
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
        if not p.get("label_recorded"):
            self._record_label(ticker, p, exit_price, timestamp)
        if self._position_open(p):
            self._close_trade(ticker, p, exit_price, timestamp, stop_result=stop_result)

    def _record_label(self, ticker: str, p: dict, label_price: float, timestamp) -> None:
        """The supervision half of resolving a prediction: drift monitor,
        outcome log, replay buffer. Split out of `_resolve_one` so a
        trailing-stop "runner" (see `_resolve_matured`) can record its
        training label at the real label horizon, exactly like every
        other prediction, while its position stays open past it --
        otherwise letting winners run would quietly change what the
        model is trained on.

        A position re-tracked at startup (_reconcile_positions) has no
        feature window / model call behind it in this process, so it
        isn't a prediction outcome: it must not feed the drift monitor's
        hit-rate or become a training sample (its realized P&L still
        counts in _close_trade -- that money is real)."""
        if p.get("feature_window") is None:
            p["label_recorded"] = True
            return
        realized_ret = (label_price - p["entry_price"]) / p["entry_price"]
        deadband = self.label_cfg.deadband_bps / 10_000.0
        true_action = 1 if realized_ret > deadband else (-1 if realized_ret < -deadband else 0)
        # Score the model's own prediction, which differs from the traded
        # action only under a backtest signal_override (see _try_predict).
        pred_action_signed = p.get("model_action", p["action"]) - 1  # {0,1,2} -> {-1,0,1}
        correct = (pred_action_signed == true_action)

        self.drift_monitor.record_prediction_outcome(correct, p.get("model_confidence", p["confidence"]))
        if self.drift_state_path is not None:
            self.drift_monitor.save_state(self.drift_state_path)
        self.logger.log_outcome(p["decision_id"], realized_ret, correct)
        self.buffer.add(p["feature_window"], true_action + 1, realized_ret, ticker, timestamp)
        self.trainer.notify_new_samples(1)
        p["label_recorded"] = True

    def _close_trade(self, ticker: str, p: dict, exit_price: float, timestamp,
                     stop_result: tuple | None = None) -> None:
        """The trading half of resolving a filled prediction: realized
        P&L into the risk manager and trade_log, and the real closing
        order at the broker. Only called for size_pct_equity > 0."""
        # Release this entry's broker-side stop BEFORE closing: it reserves
        # the shares (Alpaca rejects a market sell of shares held for
        # another order), and if it already EXECUTED at the broker the
        # position is gone -- a second closing order would open a position
        # in the opposite direction. In that case book the exit at the
        # broker's real stop fill price instead.
        stop_status, stop_fill = (stop_result if stop_result is not None
                                  else self._release_protective_stop(ticker, p, timestamp))
        if stop_status == "filled" and stop_fill is not None and stop_fill.price > 0:
            exit_price = stop_fill.price
        realized_ret = (exit_price - p["entry_price"]) / p["entry_price"]
        pred_action_signed = p["action"] - 1  # {0,1,2} -> {-1,0,1}

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
        pnl_pct_of_equity = p["size_pct_equity"] * pred_action_signed * realized_ret
        # update_after_trade_result() can trip the kill switch (a
        # daily-loss or consecutive-loss breach) all on its own --
        # but, unlike the DriftMonitor-triggered halt a few hundred
        # lines below in run() (which always pairs trip_kill_switch()
        # with self.logger.log_halt()), nothing here ever logged a
        # "halt" event for THIS trip path. The kill switch was still
        # correctly enforced (size_order() checks
        # kill_switch_engaged() on every call, so no trade kept
        # placing), but decisions.jsonl never recorded that the trip
        # happened -- so a dashboard computing its ARMED/HALTED badge
        # purely from logged halt/kill_switch_reset events (see
        # dashboard.html's haltedNow) could show ARMED while the
        # kill switch was genuinely engaged, directly contradicting
        # the Decision Feed, which reads risk_manager's real-time
        # "kill switch engaged" reason on every subsequent sizing
        # decision. Diffing halt_reasons before/after (rather than
        # just checking kill_switch_engaged() before vs. after, the
        # way the DriftMonitor call site does) also catches the case
        # where a single call trips BOTH the daily-loss and
        # consecutive-loss checks at once, or trips again while
        # already engaged (e.g. "6 consecutive losing trades" then
        # "7" on the next loss) -- each new reason is still real
        # information worth putting in the halt history.
        reasons_before = len(self.risk_manager.state.halt_reasons)
        self.risk_manager.update_after_trade_result(pnl_pct_of_equity)
        new_halt_reasons = self.risk_manager.state.halt_reasons[reasons_before:]
        if new_halt_reasons:
            self.logger.log_halt(list(new_halt_reasons))
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
            "direction": pred_action_signed,
            "entry_timestamp": p.get("entry_timestamp"),
            "confidence": p.get("confidence"),
            "expected_return": p.get("expected_return"),
            "realized_return": realized_ret * pred_action_signed,
        })

        # Actually flatten the real broker position this prediction's
        # entry opened -- before this, NOTHING below this point ever
        # existed: both exit paths (natural maturity and the hard
        # stop-loss) only ever updated the synthetic bookkeeping
        # above (drift monitor, replay buffer, trade_log, risk
        # manager counters). broker.positions/.cash were touched
        # exclusively by NEW entry orders in run()'s submit_order
        # call (confirmed by grep: exactly one call site, only for
        # new signals) -- so a "closed" position just kept sitting
        # open at the real broker, continuing to drift with the
        # market, until some unrelated future signal happened to net
        # it down. `p["filled_quantity"]` is the exact share count
        # this entry's own Fill reported, so the closing order here
        # flattens precisely this entry's contribution, leaving any
        # other still-open entry in the same ticker's FIFO queue
        # untouched (see TestRealBrokerCloseOnResolution in
        # test_orchestrator.py for the overlapping-entries case).
        if stop_status == "filled":
            # Already flat at the broker -- record the broker's own stop
            # fill as this entry's close; submit nothing.
            if stop_fill is not None:
                self.logger.log_fill(p["decision_id"], stop_fill)
        else:
            close_qty = p["filled_quantity"]
            if stop_status == "unknown" or p.get("reconciled"):
                # Couldn't confirm the stop's fate (or this entry was
                # re-tracked at startup, where the saved record may be
                # stale): never close more than the broker still holds in
                # this direction, so this closing order can never open a
                # new position the other way.
                direction = 1 if Action(p["action"]) == Action.BUY else -1
                held = self.broker.get_position(ticker).quantity * direction
                close_qty = max(0.0, min(close_qty, held))
            close_action = Action.SELL if Action(p["action"]) == Action.BUY else Action.BUY
            close_fill = None
            if close_qty > 0:
                close_fill = self.broker.close_quantity(
                    ticker, close_action, close_qty, exit_price, timestamp
                )
            self.logger.log_close_order(p["decision_id"], ticker, close_qty, close_fill)
            if close_fill is not None:
                self.logger.log_fill(p["decision_id"], close_fill)
        p["position_closed"] = True
        # _save_positions skips closed positions, so this is correct even
        # though p may stay in _pending until its label is recorded.
        self._save_positions()

    # -- protective stops & position persistence ---------------------------

    def _place_protective_stop(self, ticker: str, p: dict) -> None:
        """Rest a broker-side stop for a FILLED entry at the same level the
        local per-bar check enforces (entry_price x (1 -/+ stop_loss_pct)),
        recording the order id on the entry as "stop_order_id". With
        trailing_stop_enabled the local trail can tighten past this level;
        the broker stop stays at the initial stop as the backstop."""
        p["stop_order_id"] = None
        if not self.protective_stops or not self._position_open(p) or not p.get("stop_loss_pct"):
            return
        direction = p["action"] - 1  # {0,1,2} -> {-1,0,1}
        if direction == 0 or p["filled_quantity"] <= 0:
            return
        stop_frac = p["stop_loss_pct"] / 100.0
        stop_price = p["entry_price"] * (1 - stop_frac if direction > 0 else 1 + stop_frac)
        close_action = Action.SELL if direction > 0 else Action.BUY
        order_id = self.broker.place_protective_stop(ticker, close_action, p["filled_quantity"], stop_price)
        p["stop_order_id"] = order_id
        if order_id is not None or type(self.broker).place_protective_stop is not Broker.place_protective_stop:
            self.logger.log_protective_stop(p["decision_id"], ticker,
                                            "placed" if order_id else "place_failed",
                                            order_id, stop_price, p["filled_quantity"])

    def _release_protective_stop(self, ticker: str, p: dict, timestamp) -> tuple[str, object]:
        order_id = p.get("stop_order_id")
        if not order_id:
            return ("none", None)
        status, fill = self.broker.cancel_protective_stop(order_id, timestamp)
        p["stop_order_id"] = None
        self.logger.log_protective_stop(p["decision_id"], ticker, status, order_id,
                                        quantity=fill.quantity if fill is not None else None)
        return (status, fill)

    def _lift_protective_stops(self, ticker: str, timestamp) -> list[dict]:
        """Temporarily cancel this ticker's resting broker stops before a
        NEW order on it. Alpaca rejects an order while an opposite-side
        order is open on the same symbol ("potential wash trade"), and
        shares reserved by a resting sell stop can't be sold by another
        order -- so without this, adding to (or trading against) a
        protected position would be rejected. Any stop found to have
        already executed closes its position right here (its label is
        still recorded at the horizon, like any early close). Returns the
        entries whose stops should be re-placed afterwards via
        _restore_protective_stops()."""
        lifted = []
        for p in list(self._pending[ticker]):
            if not p.get("stop_order_id"):
                continue
            order_id = p["stop_order_id"]
            status, fill = self._release_protective_stop(ticker, p, timestamp)
            if status == "filled" and fill is not None:
                self._close_trade(ticker, p, fill.price, timestamp, stop_result=(status, fill))
            elif status == "unknown":
                # The cancel wasn't confirmed: the stop may still be
                # resting, or partly filled. Placing a second full-size
                # stop could double it up (two BUY stops on a short can
                # flip it long), so keep the handle and don't re-place.
                p["stop_order_id"] = order_id
            else:
                lifted.append(p)
        self._pending[ticker] = deque(p for p in self._pending[ticker] if not self._finished(p))
        return lifted

    def _restore_protective_stops(self, ticker: str, lifted: list[dict]) -> None:
        still_open = {id(e) for e in self._pending[ticker] if self._position_open(e)}
        for p in lifted:
            if id(p) in still_open and not p.get("stop_order_id"):
                self._place_protective_stop(ticker, p)
        if lifted:
            self._save_positions()

    _PERSISTED_KEYS = ("decision_id", "entry_price", "action", "confidence", "size_pct_equity",
                       "filled_quantity", "stop_loss_pct", "stop_order_id")

    def _save_positions(self) -> None:
        """Persist every open FILLED position (everything needed to keep
        managing it: entry, direction, quantity, stop, broker stop id, bars
        left until its time exit and bars held so far) -- not the feature
        window, which is large and only matters for the training label.
        Without this, a restart started with an empty _pending and every
        position already held at the broker was orphaned: no stop-loss, no
        time exit, and (while halted) no way to sell it at all."""
        if self.positions_state_path is None:
            return
        out = {}
        for t, pending in self._pending.items():
            rows = []
            for p in pending:
                if not self._position_open(p):
                    continue
                row = {k: p.get(k) for k in self._PERSISTED_KEYS}
                row["remaining_bars"] = int(max(0, p["mature_at_count"] - self._bar_count[t]))
                if p.get("entry_count") is not None:
                    row["bars_held"] = int(max(0, self._bar_count[t] - p["entry_count"]))
                rows.append(row)
            if rows:
                out[t] = rows
        path = Path(self.positions_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        text = json.dumps(out, indent=2)
        try:
            # Write-then-rename so a crash mid-write never leaves a
            # truncated file that would make the next startup adopt
            # everything from scratch.
            tmp.write_text(text)
            tmp.replace(path)
        except OSError:
            path.write_text(text)  # e.g. a bind mount that refuses rename

    def _load_saved_positions(self) -> dict:
        if self.positions_state_path is None or not Path(self.positions_state_path).is_file():
            return {}
        try:
            data = json.loads(Path(self.positions_state_path).read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError) as exc:
            print(f"[Orchestrator] could not read {self.positions_state_path}: {exc!r} -- "
                  f"treating every broker position as untracked.")
            return {}

    def _current_realized_vol(self, ticker: str) -> float:
        hist = self._history.get(ticker)
        if hist is None or len(hist) < 20:
            return float("nan")
        try:
            v = compute_features(hist)["realized_vol_15"].iloc[-1]
            return float(v)
        except Exception:  # noqa: BLE001 -- fall back to the min stop, never crash startup
            return float("nan")

    def _reconcile_positions(self) -> None:
        """Re-track, at startup, every position the broker actually holds
        for our tickers, so none is left without a stop-loss or time exit.

        The broker's position is the source of truth for WHAT is held;
        open_positions.json (if any) supplies HOW to manage it (original
        entry, stop, bars left). Per ticker:
          - broker flat: drop any saved entries (closed while we were
            down, e.g. by a broker-side stop).
          - saved entries in the held direction cover the broker quantity:
            resume them, with their remaining horizon.
          - broker holds more than saved entries cover (or nothing was
            saved): adopt the untracked remainder as one entry at the
            broker's average entry price, with a fresh volatility-scaled
            stop and a full horizon (adopted_horizon_bars).
          - saved entries claim MORE than the broker holds: they're stale;
            adopt the broker quantity as one entry instead.
        Stale broker stops from the previous run are canceled first, then
        a fresh protective stop is placed for every tracked entry (DAY
        stops expire overnight). Re-tracked entries carry no training
        label (label_recorded is set up front). Runs regardless of the
        kill switch: managing an existing position is never "new risk"."""
        saved = self._load_saved_positions()
        for rows in saved.values():
            for row in rows or []:
                if row.get("stop_order_id"):
                    self.broker.cancel_protective_stop(row["stop_order_id"], None)

        try:
            equity = float(self.broker.get_equity())
        except Exception:  # noqa: BLE001
            equity = 0.0
        if not (equity > 0):
            equity = self.risk_manager.cfg.account_equity
        horizon = self.adopted_horizon_bars if self.adopted_horizon_bars is not None else self.label_cfg.horizon

        for t in self.tickers:
            rows = list(saved.get(t) or [])
            try:
                pos = self.broker.get_position(t)
            except Exception as exc:  # noqa: BLE001
                print(f"[Orchestrator] _reconcile_positions({t}): get_position raised {exc!r}.")
                pos = Position(ticker=t, unknown=True)
            if getattr(pos, "unknown", False):
                # Couldn't ask the broker. Keep managing what was saved
                # (with fresh stops, since the old ones were just
                # canceled) rather than treating the ticker as flat and
                # deleting the only record of a position that may be open.
                if not rows:
                    continue
                print(f"[Orchestrator] _reconcile_positions({t}): broker position unknown; "
                      f"keeping {len(rows)} saved entr{'y' if len(rows) == 1 else 'ies'} as saved.")
                d0 = int(rows[0].get("action", 1)) - 1
                qty = d0 * sum(float(r.get("filled_quantity") or 0) for r in rows
                               if int(r.get("action", 1)) - 1 == d0)
                pos = Position(ticker=t, quantity=qty, avg_price=0.0)
            qty = float(pos.quantity or 0.0)
            if abs(qty) < 1e-6:
                if rows:
                    print(f"[Orchestrator] _reconcile_positions({t}): broker is flat; dropping "
                          f"{len(rows)} saved entr{'y' if len(rows) == 1 else 'ies'} (closed while offline).")
                continue
            direction = 1 if qty > 0 else -1
            held = abs(qty)
            rows = [r for r in rows if (int(r.get("action", 1)) - 1) == direction
                    and float(r.get("filled_quantity") or 0) > 0]
            tracked = sum(float(r["filled_quantity"]) for r in rows)
            tol = max(1e-3, 0.01 * held)
            if tracked > held + tol:
                rows, tracked = [], 0.0

            count = self._bar_count[t]
            entries = []
            for r in rows:
                entries.append(({
                    "decision_id": r.get("decision_id") or f"resumed-{uuid.uuid4().hex[:12]}",
                    "entry_price": float(r["entry_price"]),
                    "action": int(r["action"]),
                    "confidence": float(r.get("confidence") or 0.0),
                    "feature_window": None,
                    "mature_at_count": count + int(r.get("remaining_bars") or 0),
                    "entry_count": count - int(r.get("bars_held") or 0),
                    "size_pct_equity": float(r["size_pct_equity"]),
                    "filled_quantity": float(r["filled_quantity"]),
                    "stop_loss_pct": r.get("stop_loss_pct") or self.risk_manager.stop_loss_pct_for(
                        self._current_realized_vol(t)),
                    "label_recorded": True,
                    "reconciled": True,
                }, "saved"))
            remainder = held - tracked
            if remainder > tol:
                entry_price = float(pos.avg_price or 0.0)
                if not (entry_price > 0) and len(self._history.get(t, [])):
                    entry_price = float(self._history[t]["close"].iloc[-1])
                if entry_price > 0:
                    entries.append(({
                        "decision_id": f"adopted-{uuid.uuid4().hex[:12]}",
                        "entry_price": entry_price,
                        "action": 2 if direction > 0 else 0,
                        "confidence": 0.0,
                        "feature_window": None,
                        "mature_at_count": count + int(horizon),
                        "entry_count": count,
                        "size_pct_equity": remainder * entry_price / equity * 100.0,
                        "filled_quantity": remainder,
                        "stop_loss_pct": self.risk_manager.stop_loss_pct_for(self._current_realized_vol(t)),
                        "label_recorded": True,
                        "reconciled": True,
                    }, "adopted"))

            for p, source in entries:
                self._place_protective_stop(t, p)
                self._pending[t].append(p)
                self.logger.log_position_reconciled(
                    t, p["decision_id"], source, p["filled_quantity"] * direction, p["entry_price"],
                    p["stop_loss_pct"], p["mature_at_count"] - count)
                print(f"[Orchestrator] _reconcile_positions({t}): {source} {p['filled_quantity']:.4f} "
                      f"{'long' if direction > 0 else 'short'} @ {p['entry_price']:.2f}, "
                      f"stop {p['stop_loss_pct']:.2f}%, time exit in "
                      f"{p['mature_at_count'] - count} bars.")
            if entries:
                # _resolve_matured pops from the left while mature_at_count
                # has passed, so keep the queue ordered by it.
                self._pending[t] = deque(sorted(self._pending[t], key=lambda e: e["mature_at_count"]))
        self._save_positions()

    def _resolve_matured(self, ticker: str, current_bar) -> None:
        cfg = self.risk_manager.cfg
        if cfg.max_hold_bars > 0:
            # An explicit hold length wins over letting trailing-stop
            # winners run: every filled position closes `max_hold_bars`
            # after entry (or earlier, at its stop).
            self._resolve_with_hold(ticker, current_bar, cfg.max_hold_bars)
            return
        if not (cfg.trailing_stop_enabled and cfg.let_winners_run):
            pending = self._pending[ticker]
            while pending and pending[0]["mature_at_count"] <= self._bar_count[ticker]:
                p = pending.popleft()
                self._resolve_one(ticker, p, current_bar.close, current_bar.timestamp)
            return

        # Letting winners run: a FILLED position whose trailing stop has
        # armed (it got at least trail_activation_mult R in profit) is
        # not closed at the label horizon. Its training label is still
        # recorded right at the horizon, same as every other prediction
        # (see _record_label), but the position stays open under its
        # trailing stop / take profit (_check_stop_losses) until one of
        # those fires or max_runner_bars have passed since entry.
        count = self._bar_count[ticker]
        survivors: deque = deque()
        for p in self._pending[ticker]:
            if p["mature_at_count"] > count:
                survivors.append(p)
                continue
            if p["size_pct_equity"] > 0 and p.get("trail_stop") is not None:
                if not p.get("label_recorded"):
                    self._record_label(ticker, p, current_bar.close, current_bar.timestamp)
                entry_count = p.get("entry_count", p["mature_at_count"] - self.label_cfg.horizon)
                if count - entry_count < cfg.max_runner_bars:
                    survivors.append(p)
                    continue
            self._resolve_one(ticker, p, current_bar.close, current_bar.timestamp)
        self._pending[ticker] = survivors

    def _resolve_with_hold(self, ticker: str, current_bar, hold: int) -> None:
        """max_hold_bars > 0: the label and the position are resolved on
        their own clocks. The training label is still recorded at the
        label horizon (so the model learns exactly what it would with the
        default exit), while a filled position is closed `hold` bars after
        entry -- earlier or later than the horizon."""
        count = self._bar_count[ticker]
        survivors: deque = deque()
        for p in self._pending[ticker]:
            if not p.get("label_recorded") and p["mature_at_count"] <= count:
                self._record_label(ticker, p, current_bar.close, current_bar.timestamp)
            if self._position_open(p):
                entry_count = p.get("entry_count", p["mature_at_count"] - self.label_cfg.horizon)
                if count - entry_count >= hold:
                    self._close_trade(ticker, p, current_bar.close, current_bar.timestamp)
            if not self._finished(p):
                survivors.append(p)
        self._pending[ticker] = survivors

    @staticmethod
    def _position_open(p: dict) -> bool:
        return p["size_pct_equity"] > 0 and not p.get("position_closed")

    def _finished(self, p: dict) -> bool:
        return bool(p.get("label_recorded")) and not self._position_open(p)

    def _close_position_early(self, ticker: str, p: dict, exit_price: float, timestamp) -> None:
        """Close a filled position before its label horizon WITHOUT
        recording its training label: the entry stays pending and its
        label is still recorded at the horizon, so an exit rule can't
        change what the model is trained on."""
        self._close_trade(ticker, p, exit_price, timestamp)

    def _exit_on_reversal(self, ticker: str, pred: dict, price: float, timestamp) -> int:
        """exit_on_reversal: a confident signal (>= min_confidence) in the
        opposite direction closes this ticker's open positions at this
        bar's close. Returns how many were closed."""
        direction = pred["action"] - 1
        if direction == 0 or pred["confidence"] < self.risk_manager.cfg.min_confidence:
            return 0
        closed = 0
        for p in self._pending[ticker]:
            if self._position_open(p) and (p["action"] - 1) == -direction:
                self._close_position_early(ticker, p, price, timestamp)
                closed += 1
        return closed

    def _update_trailing_stop(self, p: dict, current_bar, direction: int, stop_frac: float) -> None:
        """Track the best price this position has seen and, once it is
        trail_activation_mult R in profit (R = the entry's own initial
        stop distance), arm a trailing stop trail_distance_mult R behind
        that best price. The trail only ever tightens, and only moves
        when it would improve by at least broker_stop_update_frac of its
        distance -- the granularity a real broker-side stop order would
        be re-sent at, so the backtest doesn't assume a stop that
        follows every tick. Applied from the NEXT bar on: this bar's
        high/low already counted toward the best price, and an OHLC bar
        can't say whether its extreme came before or after its low/high,
        so tightening the stop within the same bar would peek ahead."""
        cfg = self.risk_manager.cfg
        entry = p["entry_price"]
        extreme = current_bar.high if direction > 0 else current_bar.low
        best = p.get("best_price", entry)
        best = max(best, extreme) if direction > 0 else min(best, extreme)
        p["best_price"] = best

        risk = entry * stop_frac
        if risk <= 0 or (best - entry) * direction < cfg.trail_activation_mult * risk:
            return
        trail_dist = cfg.trail_distance_mult * risk
        candidate = best - direction * trail_dist
        current = p.get("trail_stop")
        if current is None:
            initial_stop = entry * (1 - direction * stop_frac)
            if (candidate - initial_stop) * direction > 0:
                p["trail_stop"] = candidate
        elif (candidate - current) * direction >= cfg.broker_stop_update_frac * trail_dist:
            p["trail_stop"] = candidate

    def _check_stop_losses(self, ticker: str, current_bar) -> None:
        """Close out, THIS bar, any pending (not-yet-matured) FILLED
        position whose intrabar high/low has breached ITS OWN stop
        level -- `p["stop_loss_pct"]`, computed once at entry time by
        `RiskManager.stop_loss_pct_for()` from that ticker's realized
        volatility right then, not one flat percentage shared by every
        ticker and every entry.

        Before the stop was enforced AT ALL, `RiskManager.size_order()`
        computed and returned a `stop_loss_pct` on every sized order,
        but nothing in `Orchestrator` ever read it (confirmed by grep):
        a filled position was only ever closed at `mature_at_count`,
        i.e. after a full `label_cfg.horizon` bars (15, by default) had
        elapsed, however far the price moved against it in the
        meantime -- the exact same dead-code-safety-check pattern as
        `reset_daily_counters`/`open_notional_pct` above, just for the
        per-position (not daily or portfolio-wide) risk limit. Once
        enforcement existed, it still used one flat `hard_stop_loss_pct`
        for every ticker regardless of how volatile that ticker actually
        was -- tight enough to be routinely triggered by ordinary noise
        on a choppy name, loose enough to barely matter on a calm one.
        `stop_loss_pct_for()`'s docstring has the concrete numbers; this
        method just reads whatever per-entry value sizing already
        computed, rather than computing (or hardcoding) a stop level
        itself.

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
        cfg = self.risk_manager.cfg
        survivors: deque = deque()
        for p in pending:
            if not self._position_open(p):
                survivors.append(p)
                continue
            stop_frac = p["stop_loss_pct"] / 100.0
            direction = p["action"] - 1  # {0,1,2} -> {-1,0,1}; filled => never 0
            exit_price = None
            # An armed trailing stop (see _update_trailing_stop) replaces
            # the initial stop -- it is only ever set tighter than it.
            trail_stop = p.get("trail_stop") if cfg.trailing_stop_enabled else None
            if direction > 0:
                stop_level = trail_stop if trail_stop is not None else p["entry_price"] * (1 - stop_frac)
                if current_bar.low <= stop_level:
                    exit_price = stop_level
            elif direction < 0:
                stop_level = trail_stop if trail_stop is not None else p["entry_price"] * (1 + stop_frac)
                if current_bar.high >= stop_level:
                    exit_price = stop_level
            # Take profit at take_profit_mult R. Checked after the stop:
            # when one OHLC bar spans both levels there's no telling which
            # was hit first, so assume the worse one.
            if exit_price is None and cfg.trailing_stop_enabled and cfg.take_profit_mult > 0:
                tp_level = p["entry_price"] * (1 + direction * cfg.take_profit_mult * stop_frac)
                if (direction > 0 and current_bar.high >= tp_level) or \
                        (direction < 0 and current_bar.low <= tp_level):
                    exit_price = tp_level
            # Fixed take profit (no trailing): closes the position but,
            # unlike the stop, leaves the label to be recorded at the horizon.
            if exit_price is None and not cfg.trailing_stop_enabled and cfg.fixed_take_profit_mult > 0:
                tp_level = p["entry_price"] * (1 + direction * cfg.fixed_take_profit_mult * stop_frac)
                if (direction > 0 and current_bar.high >= tp_level) or \
                        (direction < 0 and current_bar.low <= tp_level):
                    self._close_position_early(ticker, p, tp_level, current_bar.timestamp)
                    if not self._finished(p):
                        survivors.append(p)
                    continue
            if exit_price is not None:
                self._resolve_one(ticker, p, exit_price, current_bar.timestamp)
            else:
                if cfg.trailing_stop_enabled:
                    self._update_trailing_stop(p, current_bar, direction, stop_frac)
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

    def _maybe_apply_external_kill_switch_reset(self) -> None:
        """Polls RiskManager.check_for_reset_request() once per bar for a
        reset request written by scripts/serve_dashboard.py's "Reset
        kill switch" button (see RiskManager.write_reset_request's
        docstring for why that round trip through a sentinel file is
        needed at all: the dashboard server is a separate process with
        no reference to this live RiskManager instance).

        Only does anything when risk_state_path is set -- the same
        directory that holds risk_state.json is where the dashboard is
        told to write its request (both scripts/run_live_alpaca.py and
        scripts/serve_dashboard.py point at checkpoints/ by default), so
        a run that isn't persisting risk state anywhere (backtests,
        tests) has no shared location for a dashboard to even write to,
        and nothing to poll here either."""
        if self.risk_state_path is None:
            return
        result = self.risk_manager.check_for_reset_request(Path(self.risk_state_path).parent)
        if result is not None:
            reasons_text = "; ".join(result["cleared_reasons"]) or "unknown reason"
            print(f"[Orchestrator] Kill switch reset via {result['source']} "
                  f"(was halted for: {reasons_text}).")
            self.logger.log_kill_switch_reset(result["cleared_reasons"], result["source"])
            self.risk_manager.save_state(self.risk_state_path)

    def _prime_history(self) -> None:
        """Warm up each ticker's rolling history from feed.get_history()
        before the main loop starts, so prediction can begin almost
        immediately instead of waiting out warmup_bars worth of live bars
        (which, on real market data, means waiting warmup_bars minutes --
        two hours at the default of 120 -- before the model says anything).
        Safe to call on any feed: SyntheticFeed/YFinanceFeed advance their
        internal cursor on this call so stream() picks up right after the
        warm-up window instead of replaying it.

        Used to swallow a failed/empty get_history() for a ticker
        completely silently -- `except Exception: hist = None` followed
        by `continue`, with no print, no log, nothing. On a live feed
        (AlpacaLiveFeed) that call can fail for reasons worth knowing
        about right away (feed/subscription permissions, a transient
        REST error, an unrecognized ticker) and the failure mode is easy
        to miss: with the REST warm-up silently skipped, that ticker
        falls back to accumulating `warmup_bars` live bars from scratch
        (two hours at the default of 120) before _try_predict() ever
        returns anything for it -- indistinguishable, from the log alone,
        from "this is just a slow ticker" or "nothing's wrong yet,
        wait". Each ticker now gets exactly one line stating what
        happened, so that distinction is visible without having to infer
        it from an absence of `prediction` events an hour or two later."""
        for t in self.tickers:
            try:
                hist = self.feed.get_history(t, self.max_history)
            except Exception as exc:  # noqa: BLE001 -- report every feed failure, not just the ones we anticipated
                print(f"[Orchestrator] _prime_history({t}): get_history() raised "
                      f"{exc!r} -- falling back to warming up from live bars only "
                      f"({self.warmup_bars} needed before the first prediction).")
                continue
            if hist is None or len(hist) == 0:
                print(f"[Orchestrator] _prime_history({t}): get_history() returned no bars "
                      f"-- falling back to warming up from live bars only "
                      f"({self.warmup_bars} needed before the first prediction).")
                continue
            self._history[t] = hist.iloc[-self.max_history:]
            self._bar_count[t] = len(self._history[t])
            status = "ready to predict immediately" if self._bar_count[t] >= self.warmup_bars                 else f"still needs {self.warmup_bars - self._bar_count[t]} more live bars"
            print(f"[Orchestrator] _prime_history({t}): warmed up with "
                  f"{self._bar_count[t]} bars ({status}).")

    # -- main loop ----------------------------------------------------------

    def run(self, max_bars: int | None = None, stop_check=None) -> None:
        """Run the loop. `stop_check`, if given, is a zero-arg callable
        polled once per bar; returning True ends the run cleanly (used for
        e.g. "stop when the market closes" in live trading, without the
        feed itself needing to know about market hours)."""
        self._prime_history()
        self._reconcile_positions()
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
            self._maybe_apply_external_kill_switch_reset()

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
                    pred.get("model_action", pred["action"]), pred.get("model_confidence", pred["confidence"]),
                    pred.get("model_expected_return", pred["expected_return"]),
                )
                if self.risk_manager.cfg.exit_on_reversal and \
                        self._exit_on_reversal(bar.ticker, pred, bar.close, bar.timestamp):
                    equity = self.broker.get_equity(mark_prices)
                    self.risk_manager.update_account_equity(equity)
                    self.risk_manager.update_open_exposure(self._gross_exposure_pct(mark_prices, equity))
                    self.risk_manager.update_per_ticker_exposure(
                        bar.ticker, self._ticker_notional_pct(bar.ticker, mark_prices, equity)
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
                    lifted = self._lift_protective_stops(bar.ticker, bar.timestamp)
                    fill = self.broker.submit_order(bar.ticker, sizing["action"], notional, bar.close, bar.timestamp)
                    self._restore_protective_stops(bar.ticker, lifted)
                    if fill is not None:
                        self.logger.log_fill(decision_id, fill)

                self._pending[bar.ticker].append({
                    "decision_id": decision_id,
                    "entry_price": bar.close,
                    "action": pred["action"],
                    "confidence": pred["confidence"],
                    "model_action": pred.get("model_action", pred["action"]),
                    "model_confidence": pred.get("model_confidence", pred["confidence"]),
                    "expected_return": pred["expected_return"],
                    "entry_timestamp": bar.timestamp,
                    "feature_window": pred["feature_window"],
                    "mature_at_count": self._bar_count[bar.ticker] + self.label_cfg.horizon,
                    # When this entry was opened -- max_runner_bars is
                    # counted from here (see _resolve_matured).
                    "entry_count": self._bar_count[bar.ticker],
                    # 0.0 unless a real order was actually filled -- see
                    # _resolve_matured()'s use of this to feed realized P&L
                    # into the risk manager. A rejected/skipped order (fill
                    # is None) must not be counted as a realized trade.
                    "size_pct_equity": sizing["size_pct_equity"] if fill is not None else 0.0,
                    # The exact share count this order's Fill reported --
                    # 0.0 if it never filled. _resolve_one uses this (not
                    # size_pct_equity, which is a dollar fraction) to
                    # submit a real closing order for precisely this
                    # entry's own contribution to the ticker's position.
                    "filled_quantity": fill.quantity if fill is not None else 0.0,
                    # This entry's own vol-scaled stop (RiskManager.
                    # stop_loss_pct_for(), computed from THIS bar's
                    # realized_vol at sizing time) -- None when sizing
                    # produced "no trade" (sizing["stop_loss_pct"] is
                    # always None in that case; _check_stop_losses never
                    # reads this for a size_pct_equity <= 0 entry anyway).
                    "stop_loss_pct": sizing["stop_loss_pct"],
                })
                if fill is not None:
                    self._place_protective_stop(bar.ticker, self._pending[bar.ticker][-1])
                    self._save_positions()

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
