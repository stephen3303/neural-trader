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
import math
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
    per_ticker_notional_pct: dict = field(default_factory=dict)


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

    def update_open_exposure(self, open_notional_pct: float) -> None:
        """Sync `state.open_notional_pct` -- the number `size_order()`'s
        portfolio-wide gross-exposure cap (`cfg.max_gross_exposure_pct`)
        checks against -- to the broker's actual current aggregate open
        position notional, as a percent of equity.

        Before this, `open_notional_pct` was initialized to 0.0 in
        `RiskState` and NEVER updated anywhere in the codebase (confirmed
        by grep -- not even a test exercised a nonzero value). That made
        the gross-exposure cap completely non-functional:
        `size_order()`'s `if self.state.open_notional_pct >=
        self.cfg.max_gross_exposure_pct` could never fire (0.0 is never
        >= a positive cap), and `max_gross_exposure_pct -
        self.state.open_notional_pct` always evaluated to the full,
        uncapped `max_gross_exposure_pct` -- so with enough tickers each
        independently sized up to `max_position_pct`, aggregate exposure
        across the whole portfolio could exceed the configured 60%
        default with nothing to stop it. `max_position_pct` (the
        per-ticker cap) was never affected by this and still worked.

        Ignores non-finite/negative values rather than corrupting state,
        the same defensive pattern as `update_account_equity` -- a bad
        reading should never silently remove the cap by reporting 0%
        exposure, or make it impossible to trade by reporting garbage."""
        if not math.isfinite(open_notional_pct) or open_notional_pct < 0:
            return
        self.state.open_notional_pct = open_notional_pct

    def update_per_ticker_exposure(self, ticker: str, notional_pct: float) -> None:
        """Sync `state.per_ticker_notional_pct[ticker]` -- the number
        `size_order()` needs to cap a NEW order to a ticker's REMAINING
        headroom under `cfg.max_position_pct`, instead of applying that
        cap fresh to every single order regardless of how much of that
        ticker is already held.

        Before this, `max_position_pct` ("single position cap, % of
        equity" -- see `RiskConfig`) was only ever checked against each
        INDIVIDUAL new order's own size, never against the ticker's
        already-accumulated position. A sustained run of same-direction
        signals on one ticker (e.g. a real trend, which is exactly the
        condition this strategy is built to ride) could therefore keep
        adding to that ticker's position bar after bar with nothing to
        stop it, driving its real exposure well past the documented
        cap. Verified directly on a real 3,000-bar/12-ticker run: before
        this fix, QQQ's position reached 17.9% of equity and NVDA's
        13.9%, both past the configured 10% cap, with every individual
        order along the way still correctly <= 10% on its own.

        Same defensive non-finite/negative handling as
        `update_open_exposure`; 0.0 is a normal reading (no position in
        that ticker) and is accepted, not treated as a bad one."""
        if not math.isfinite(notional_pct) or notional_pct < 0:
            return
        self.state.per_ticker_notional_pct[ticker] = notional_pct

    def update_account_equity(self, equity: float) -> None:
        """Sync `cfg.account_equity` -- the number position sizing
        converts `size_pct_equity` into a real dollar notional with -- to
        the broker's actual current equity.

        Before this, `cfg.account_equity` was whatever static number was
        in config.yaml at startup, forever: for `AlpacaBroker` that's the
        real paper-account equity at the moment the process started,
        never updated again even though `AlpacaBroker.get_equity()`
        already queries Alpaca's real, current balance every bar. Any
        drift between that static assumption and the real balance (from
        deposits/withdrawals, or simply the account compounding gains or
        losses over time) silently mis-sizes every subsequent order --
        e.g. a 5%-of-equity order gets computed against a number that no
        longer matches the real account.

        Ignores non-positive/non-finite values rather than raising or
        corrupting state -- a single bad read (a transient API hiccup,
        e.g.) should never be allowed to zero out or blow up position
        sizing for the rest of the session; it just keeps using the last
        known-good equity until a valid one arrives."""
        if not math.isfinite(equity) or equity <= 0:
            return
        self.cfg.account_equity = equity

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
        # .get(..., {}) rather than data["..."]: a state file saved before
        # this field existed (pre-dating update_per_ticker_exposure) has
        # no such key -- treat that as "nothing tracked yet" rather than
        # raising KeyError on an otherwise-valid resume.
        self.state.per_ticker_notional_pct = dict(data.get("per_ticker_notional_pct", {}))
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

        # Per-ticker headroom under max_position_pct -- see
        # update_per_ticker_exposure()'s docstring for the bug this
        # closes: without this, max_position_pct was only ever checked
        # against each new order in isolation, never against how much of
        # this ticker was already held, so a sustained run of
        # same-direction signals could drive one ticker's real exposure
        # well past the documented cap.
        current_ticker_pct = self.state.per_ticker_notional_pct.get(signal.ticker, 0.0)
        if current_ticker_pct >= self.cfg.max_position_pct:
            return self._no_trade(signal, "max position pct reached for this ticker")

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
        size_pct = max(0.0, min(size_pct, self.cfg.max_position_pct - current_ticker_pct))
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
