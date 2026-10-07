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
