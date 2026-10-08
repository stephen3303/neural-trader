"""
Risk management: turns a model `Signal` into a sized order (or no order),
and owns the kill switch.

This is the layer that matters most for "eventually live trading" -- a
neural network that is simply wrong sometimes is normal and expected; what
makes a system safe to run with real money is that *this* layer never lets
a single bad prediction, a confidence spike, or a quiet drift in model
quality translate into an unbounded loss. Every check here is intentionally
conservative and fails closed (defaults to not trading) rather than open.

Position sizing uses volatility targeting (size inversely proportional to
the asset's recent realized volatility, so a calm stock and a wild stock
contribute similar risk) scaled by model confidence, then capped by a
fractional-Kelly ceiling and hard per-ticker / portfolio limits. Fractional
Kelly (not full Kelly) because full Kelly sizing is extremely sensitive to
edge misestimation -- and a continually-retrained model's estimate of its
own edge is exactly the kind of number you should not fully trust.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.model.signals import Action, Signal


@dataclass
class RiskConfig:
    account_equity: float = 100_000.0
    target_daily_vol_pct: float = 0.5      # risk ~0.5% of equity per full-conviction trade
    kelly_fraction: float = 0.25           # fraction of "full Kelly" implied by model edge
    max_position_pct: float = 10.0         # single position cap, % of equity
    max_gross_exposure_pct: float = 60.0   # sum of all open position notional, % of equity
    min_confidence: float = 0.45           # below this, treat as "hold" regardless of action
    hard_stop_loss_pct: float = 3.0        # per-position stop loss
    max_daily_loss_pct: float = 4.0        # trips the kill switch for the rest of the day
    max_consecutive_losses: int = 6        # trips the kill switch regardless of $ amount


@dataclass
class RiskState:
    trading_enabled: bool = True
    halt_reasons: list = field(default_factory=list)
    daily_pnl_pct: float = 0.0
    consecutive_losses: int = 0
    open_notional_pct: float = 0.0


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.state = RiskState()

    def kill_switch_engaged(self) -> bool:
        return not self.state.trading_enabled

    def trip_kill_switch(self, reason: str) -> None:
        self.state.trading_enabled = False
        self.state.halt_reasons.append(reason)

    def reset_kill_switch(self, *, human_confirmed: bool) -> None:
        """Deliberately requires an explicit flag -- this should only ever
        be called from a human-operated override (e.g. a dashboard button),
        never automatically by the trading loop itself."""
        if not human_confirmed:
            raise PermissionError("Kill switch reset requires explicit human confirmation.")
        self.state.trading_enabled = True
        self.state.halt_reasons.clear()
        self.state.consecutive_losses = 0

    def update_after_trade_result(self, pnl_pct_of_equity: float) -> None:
        self.state.daily_pnl_pct += pnl_pct_of_equity
        if pnl_pct_of_equity < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        if self.state.daily_pnl_pct <= -self.cfg.max_daily_loss_pct:
            self.trip_kill_switch(
                f"daily loss {self.state.daily_pnl_pct:.2f}% breached max "
                f"{self.cfg.max_daily_loss_pct:.2f}%"
            )
        if self.state.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.trip_kill_switch(
                f"{self.state.consecutive_losses} consecutive losing trades"
            )

    def reset_daily_counters(self) -> None:
        self.state.daily_pnl_pct = 0.0

    def save_state(self, path: str | Path) -> None:
        """Persist RiskState (trading_enabled, halt_reasons, daily_pnl_pct,
        consecutive_losses, open_notional_pct) so a process restart doesn't
        silently forget today's drawdown/loss-streak and resume trading as
        if the kill switch had never been under pressure. Called by
        Orchestrator after every state-changing event, not just on clean
        shutdown -- a crash should lose at most one event's worth of
        state, not the whole day."""
        Path(path).write_text(json.dumps(asdict(self.state), indent=2))

    def load_state(self, path: str | Path) -> bool:
        """Restore RiskState from `save_state`'s output IN PLACE
        (self.state is mutated, never rebound) so this is safe to call
        before or after anything else holds a reference to self.state.
        Returns True if a state file existed and was loaded, False if
        there was nothing to resume from (e.g. first-ever run)."""
        p = Path(path)
        if not p.is_file():
            return False
        data = json.loads(p.read_text())
        self.state.trading_enabled = data["trading_enabled"]
        self.state.halt_reasons = list(data["halt_reasons"])
        self.state.daily_pnl_pct = data["daily_pnl_pct"]
        self.state.consecutive_losses = data["consecutive_losses"]
        self.state.open_notional_pct = data["open_notional_pct"]
        return True

    def size_order(self, signal: Signal, realized_vol: float) -> dict:
        """Returns a dict describing the sizing decision. `size_pct` is the
        recommended position size as a percent of account equity (0 means
        no trade). `realized_vol` is the asset's recent return volatility
        (e.g. from features.realized_vol_15), used to normalize risk across
        tickers of different "jumpiness"."""
        if self.kill_switch_engaged():
            return self._no_trade(signal, "kill switch engaged")

        if signal.action == Action.HOLD or signal.confidence < self.cfg.min_confidence:
            return self._no_trade(signal, "hold or low confidence")

        if self.state.open_notional_pct >= self.cfg.max_gross_exposure_pct:
            return self._no_trade(signal, "max gross exposure reached")

        vol = max(realized_vol, 1e-4)  # avoid division blow-up on near-zero vol
        # Volatility-targeted base size: smaller positions in choppier names.
        base_size_pct = self.cfg.target_daily_vol_pct / vol

        # Scale by model conviction: confidence above the minimum threshold,
        # normalized to [0, 1], and the magnitude of the predicted return as
        # a very rough edge proxy for a capped-Kelly-style haircut.
        conviction = (signal.confidence - self.cfg.min_confidence) / (1 - self.cfg.min_confidence)
        conviction = max(0.0, min(1.0, conviction))
        edge_adj = min(abs(signal.expected_return) / vol, 1.0)  # crude signal-to-noise cap

        size_pct = base_size_pct * conviction * edge_adj * self.cfg.kelly_fraction
        size_pct = max(0.0, min(size_pct, self.cfg.max_position_pct))
        size_pct = min(size_pct, self.cfg.max_gross_exposure_pct - self.state.open_notional_pct)

        if size_pct <= 0:
            return self._no_trade(signal, "sized to zero after risk caps")

        return {
            "ticker": signal.ticker,
            "action": signal.action,
            "size_pct_equity": size_pct,
            "stop_loss_pct": self.cfg.hard_stop_loss_pct,
            "reason": "sized",
            "model_version": signal.model_version,
        }

    @staticmethod
    def _no_trade(signal: Signal, reason: str) -> dict:
        return {
            "ticker": signal.ticker,
            "action": Action.HOLD,
            "size_pct_equity": 0.0,
            "stop_loss_pct": None,
            "reason": reason,
            "model_version": signal.model_version,
        }
