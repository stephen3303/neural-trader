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
    5.  Check whether any *earlier* predictions have now matured (their
        label horizon has elapsed): if so, compute the realized outcome,
        feed it to the drift monitor and the replay buffer (this is the
        "continuous supervision" loop), and log it.
    6.  Update the equity curve, check the kill switch, and periodically
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
                 max_history: int = 400, device: str = "cpu", warmup_bars: int = 120):
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

        self.buffer = ReplayBuffer(window=window, n_features=model.cfg.n_features)
        self._history: dict[str, pd.DataFrame] = {t: pd.DataFrame(columns=["open", "high", "low", "close", "volume"]) for t in self.tickers}
        self._bar_count: dict[str, int] = defaultdict(int)
        self._pending: dict[str, deque] = {t: deque() for t in self.tickers}  # per-ticker FIFO of open predictions
        self.retrain_events: list[dict] = []

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

    def _resolve_matured(self, ticker: str, current_bar) -> None:
        pending = self._pending[ticker]
        while pending and pending[0]["mature_at_count"] <= self._bar_count[ticker]:
            p = pending.popleft()
            realized_ret = (current_bar.close - p["entry_price"]) / p["entry_price"]
            deadband = self.label_cfg.deadband_bps / 10_000.0
            true_action = 1 if realized_ret > deadband else (-1 if realized_ret < -deadband else 0)
            pred_action_signed = p["action"] - 1  # {0,1,2} -> {-1,0,1}
            correct = (pred_action_signed == true_action)

            self.drift_monitor.record_prediction_outcome(correct, p["confidence"])
            self.logger.log_outcome(p["decision_id"], realized_ret, correct)
            self.buffer.add(p["feature_window"], true_action + 1, realized_ret, ticker, current_bar.timestamp)
            self.trainer.notify_new_samples(1)

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

            self._resolve_matured(bar.ticker, bar)

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
                })

            equity = self.broker.get_equity(mark_prices)
            self.drift_monitor.record_equity(equity)
            self.logger.log_equity(bar.timestamp, equity)
            halted, reasons = self.drift_monitor.should_halt()
            if halted and not self.risk_manager.kill_switch_engaged():
                self.risk_manager.trip_kill_switch("; ".join(reasons))
                self.logger.log_halt(reasons)

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
