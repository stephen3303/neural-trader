"""
Live performance / concept-drift monitor.

The continual trainer (trainer.py) handles *adapting* the model to new
data. This module handles noticing when something is going wrong *faster*
than a retrain cycle can fix -- a regime shift the model hasn't adapted to
yet, a calibration breakdown, or a live drawdown -- and recommending a
trading halt rather than letting a degrading model keep sizing real
orders. Treat this as the "supervisor" that watches the model, separate
from the model itself.

Three independent signals are tracked, because any one of them alone can
be misleading:

- Rolling directional hit-rate vs. a naive baseline (is the model still
  beating "always predict hold" / coin-flip on the samples it called
  buy/sell?).
- Calibration (Brier score): are the model's confidence values still
  meaningful, or is it becoming overconfident on calls it's getting wrong?
- Realized P&L drawdown: the only metric that actually matters financially
  -- the other two can look fine while this one is bleeding out from
  costs/slippage the model doesn't see.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DriftConfig:
    window: int = 200
    min_samples: int = 50
    hit_rate_floor: float = 0.34       # 3-class baseline is 1/3; alarm if we're not clearly beating it
    brier_ceiling: float = 0.6
    max_drawdown_pct: float = 8.0      # halt trading if rolling drawdown exceeds this


@dataclass
class DriftMonitor:
    cfg: DriftConfig = field(default_factory=DriftConfig)
    _outcomes: deque = field(default_factory=deque, init=False)   # (correct: bool, confidence, pnl)
    _equity_curve: deque = field(default_factory=deque, init=False)
    _peak_equity: float = field(default=0.0, init=False)

    def __post_init__(self):
        self._outcomes = deque(maxlen=self.cfg.window)
        self._equity_curve = deque(maxlen=self.cfg.window * 5)

    def record_prediction_outcome(self, correct: bool, confidence: float) -> None:
        self._outcomes.append((correct, confidence))

    def record_equity(self, equity: float) -> None:
        self._equity_curve.append(equity)
        self._peak_equity = max(self._peak_equity, equity)

    def hit_rate(self) -> float | None:
        if len(self._outcomes) < self.cfg.min_samples:
            return None
        return sum(1 for c, _ in self._outcomes if c) / len(self._outcomes)

    def brier_score(self) -> float | None:
        if len(self._outcomes) < self.cfg.min_samples:
            return None
        # Brier score of the "was the top call correct" probability
        return sum((conf - (1.0 if correct else 0.0)) ** 2 for correct, conf in self._outcomes) / len(self._outcomes)

    def drawdown_pct(self) -> float | None:
        if not self._equity_curve or self._peak_equity <= 0:
            return None
        current = self._equity_curve[-1]
        return (self._peak_equity - current) / self._peak_equity * 100.0

    def should_halt(self) -> tuple[bool, list[str]]:
        reasons = []
        hr = self.hit_rate()
        if hr is not None and hr < self.cfg.hit_rate_floor:
            reasons.append(f"hit-rate {hr:.2%} below floor {self.cfg.hit_rate_floor:.2%}")

        brier = self.brier_score()
        if brier is not None and brier > self.cfg.brier_ceiling:
            reasons.append(f"brier score {brier:.3f} above ceiling {self.cfg.brier_ceiling:.3f} (confidence miscalibrated)")

        dd = self.drawdown_pct()
        if dd is not None and dd > self.cfg.max_drawdown_pct:
            reasons.append(f"drawdown {dd:.1f}% exceeds max {self.cfg.max_drawdown_pct:.1f}%")

        return (len(reasons) > 0, reasons)

    def snapshot(self) -> dict:
        return {
            "hit_rate": self.hit_rate(),
            "brier_score": self.brier_score(),
            "drawdown_pct": self.drawdown_pct(),
            "n_outcomes": len(self._outcomes),
        }

    def save_state(self, path: str | Path) -> None:
        """Persist the rolling outcome/equity windows and peak equity so a
        process restart doesn't silently reset hit-rate/calibration/
        drawdown tracking to empty -- which would mean should_halt() stays
        quiet for cfg.min_samples worth of fresh data after every restart,
        exactly when a system that crashed mid-session most needs it to
        still be watching. Called by Orchestrator after every
        state-changing event, not just on clean shutdown."""
        data = {
            "outcomes": [[bool(c), float(conf)] for c, conf in self._outcomes],
            "equity_curve": [float(e) for e in self._equity_curve],
            "peak_equity": self._peak_equity,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_state(self, path: str | Path) -> bool:
        """Restore from `save_state`'s output IN PLACE (mutates the
        existing deques/attribute, never rebinds self). Returns True if a
        state file existed and was loaded, False otherwise (e.g.
        first-ever run). Respects this instance's configured maxlen --
        if cfg.window shrank since the state was saved, only the most
        recent entries that still fit are kept."""
        p = Path(path)
        if not p.is_file():
            return False
        data = json.loads(p.read_text())
        self._outcomes.clear()
        for correct, conf in data["outcomes"]:
            self._outcomes.append((correct, conf))
        self._equity_curve.clear()
        for e in data["equity_curve"]:
            self._equity_curve.append(e)
        self._peak_equity = data["peak_equity"]
        return True
