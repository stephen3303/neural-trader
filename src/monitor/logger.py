"""
Structured decision logging.

Every prediction, risk decision, fill, retrain event, and realized outcome
is written as one JSON line to an append-only log. This serves two
purposes at once:

1. Observability -- a human (or a future dashboard, same spirit as the
   monitoring website in the agentic-trading-firm project) can reconstruct
   exactly why any order was or wasn't placed: what the model predicted,
   how confident it was, what the risk manager decided, and what actually
   happened.
2. Training data provenance -- because every prediction is logged with a
   `decision_id` and every realized outcome is logged separately with a
   `ref_decision_id`, you always have an auditable trail from "the model
   saw X and predicted Y" to "the label it was later trained on was Z",
   which matters a lot once a kill switch trips and you need to figure out
   whether the model or the risk layer caused the problem.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class DecisionLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict) -> None:
        record.setdefault("logged_at", time.time())
        with self.log_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_prediction(self, ticker: str, timestamp, model_version: str,
                        action: int, confidence: float, expected_return: float) -> str:
        decision_id = str(uuid.uuid4())
        self._write({
            "type": "prediction",
            "decision_id": decision_id,
            "ticker": ticker,
            "timestamp": timestamp,
            "model_version": model_version,
            "action": action,
            "confidence": confidence,
            "expected_return": expected_return,
        })
        return decision_id

    def log_risk_decision(self, decision_id: str, sizing: dict) -> None:
        self._write({"type": "risk_decision", "ref_decision_id": decision_id, **sizing})

    def log_fill(self, decision_id: str, fill) -> None:
        self._write({
            "type": "fill",
            "ref_decision_id": decision_id,
            "ticker": fill.ticker,
            "action": int(fill.action),
            "quantity": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
            "timestamp": fill.timestamp,
        })

    def log_outcome(self, decision_id: str, realized_return: float, correct: bool) -> None:
        self._write({
            "type": "outcome",
            "ref_decision_id": decision_id,
            "realized_return": realized_return,
            "correct": correct,
        })

    def log_retrain(self, record: dict) -> None:
        self._write({"type": "retrain", **record})

    def log_halt(self, reasons: list[str]) -> None:
        self._write({"type": "halt", "reasons": reasons})

    def log_kill_switch_reset(self, cleared_reasons: list[str], source: str) -> None:
        """Logs an explicit, human-confirmed kill-switch reset (see
        RiskManager.reset_kill_switch/check_for_reset_request) as its own
        event type, distinct from "halt" -- added together with the
        dashboard's "Reset kill switch" button. Without this, a
        dashboard computing HALTED/ARMED from "is there ever a halt
        record in this log" (the only signal that existed before) would
        stay stuck on HALTED forever after the very first halt, even
        once the kill switch has genuinely been reset -- this event is
        what lets it tell "halted, still active" apart from "was halted,
        since reset"."""
        self._write({"type": "kill_switch_reset", "cleared_reasons": cleared_reasons, "source": source})

    def log_equity(self, timestamp, equity: float) -> None:
        """One row per bar of total account equity (cash + marked positions).
        This is what a monitoring dashboard needs to draw an equity curve
        and compute drawdown -- without it, the log only has point-in-time
        fills, not the continuous equity path between them."""
        self._write({"type": "equity", "timestamp": timestamp, "equity": equity})

    def log_bar(self, ticker: str, timestamp, close: float) -> None:
        """One row per bar per ticker of the observed close price, so a
        dashboard can plot price alongside the model's buy/sell markers
        without needing access to the original market data feed."""
        self._write({"type": "bar", "ticker": ticker, "timestamp": timestamp, "close": close})

    def log_close_order(self, decision_id: str, ticker: str, requested_quantity: float, fill) -> None:
        """Logs whether a pending prediction's exit (label-horizon
        maturity, or an early hard stop-loss) actually flattened the real
        broker position it represents, not just the synthetic bookkeeping
        (trade_log / risk manager counters / training label) that
        Orchestrator._resolve_one always updates regardless. `fill` is
        whatever Broker.close_quantity returned: a real Fill if the
        closing order executed, or None if it was rejected or (for
        AlpacaBroker) didn't confirm within the poll window.

        Added together with Broker.close_quantity itself -- before this,
        _resolve_one never submitted any closing order at all (confirmed
        by grep: submit_order had exactly one call site in
        orchestrator.py, and it only ever fired for new signals), so the
        real position just kept drifting past every "exit" this system
        thought it had taken. This makes a close that still doesn't
        happen for real (the `fill is None` case) auditable instead of
        silently indistinguishable from one that did."""
        self._write({
            "type": "close_order",
            "ref_decision_id": decision_id,
            "ticker": ticker,
            "requested_quantity": requested_quantity,
            "filled": fill is not None,
            "filled_quantity": fill.quantity if fill is not None else 0.0,
            "filled_price": fill.price if fill is not None else None,
        })

    def log_risk_state(self, timestamp, open_notional_pct: float, per_ticker_notional_pct: dict,
                        daily_pnl_pct: float, max_gross_exposure_pct: float, max_position_pct: float,
                        max_daily_loss_pct: float) -> None:
        """One row per bar of the risk manager's live exposure/P&L state,
        alongside the configured caps it's checked against.

        Added together with the fixes that made open_notional_pct,
        per_ticker_notional_pct, and daily_pnl_pct actually meaningful
        (see RiskManager.update_open_exposure/update_per_ticker_exposure/
        reset_daily_counters' docstrings, and the README sections on
        each) -- before those fixes there was nothing worth logging here,
        since every one of these values was either permanently 0.0 or an
        uncapped running total. Lets a dashboard show that the configured
        risk caps are actually being respected over time, not just that
        the strategy made or lost money -- a safety-oriented view
        alongside the existing performance ones."""
        self._write({
            "type": "risk_state",
            "timestamp": timestamp,
            "open_notional_pct": open_notional_pct,
            "per_ticker_notional_pct": dict(per_ticker_notional_pct),
            "daily_pnl_pct": daily_pnl_pct,
            "max_gross_exposure_pct": max_gross_exposure_pct,
            "max_position_pct": max_position_pct,
            "max_daily_loss_pct": max_daily_loss_pct,
        })
