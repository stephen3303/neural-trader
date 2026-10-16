"""
Order execution. Abstract `Broker` interface so paper trading and (later)
live trading share the exact same orchestrator code path -- the only thing
that changes when you're ready to go live is which Broker class gets
constructed.

`PaperBroker` simulates fills with a simple slippage + commission model so
backtests/paper runs aren't unrealistically optimistic. `LiveBrokerStub`
is intentionally unimplemented: it documents exactly what a real
integration (Alpaca, Interactive Brokers, etc.) needs to provide, without
pretending to be one. Do not point this at a funded account until it's
backed by a real, tested broker SDK integration and has run clean in
paper mode for an extended period.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from src.model.signals import Action


@dataclass
class Fill:
    ticker: str
    action: Action
    quantity: float
    price: float
    commission: float
    timestamp: object


@dataclass
class Position:
    ticker: str
    quantity: float = 0.0
    avg_price: float = 0.0
    # True when the broker couldn't be asked (network/API error), so the
    # zero quantity means "don't know", not "flat". Callers that would
    # discard state on a flat reading must check this first.
    unknown: bool = False


class Broker(abc.ABC):
    @abc.abstractmethod
    def submit_order(self, ticker: str, action: Action, notional: float, ref_price: float, timestamp) -> Fill | None:
        """Submit an order sized by `notional` dollars at (approximately)
        `ref_price`. Returns the resulting Fill, or None if the order was
        rejected/skipped (e.g. action is HOLD). For a NEW/ADDED position,
        where sizing naturally starts from "what percent of equity" and
        only converts to a share count as an implementation detail."""

    @abc.abstractmethod
    def close_quantity(self, ticker: str, action: Action, quantity: float, ref_price: float, timestamp) -> Fill | None:
        """Submit an order for an EXACT quantity of shares/units -- not a
        dollar notional -- at (approximately) `ref_price`. Returns the
        resulting Fill, or None if it was rejected/skipped/didn't confirm.

        This exists specifically for Orchestrator._resolve_one to actually
        flatten the real position a pending prediction's exit (label-
        horizon maturity or hard stop-loss) represents. Before this
        method existed, there was no way to close a *known number of
        shares* through the Broker interface at all -- submit_order only
        ever takes a dollar notional and converts it to a quantity
        internally (via its own fill price, which includes slippage for
        PaperBroker), so re-deriving "the same notional that would, after
        slippage, buy back exactly this many shares" is fragile and
        broker-implementation-specific. `close_quantity` is the direct,
        unambiguous alternative: you already know the exact quantity to
        flatten (it's whatever the original entry's Fill reported), so
        hand that number straight to the broker instead of working
        backwards from dollars.

        `action` is the direction of THIS closing order (opposite of the
        original entry's action: closing a BUY means submitting a SELL,
        and vice versa) -- the caller decides direction, this method just
        executes it."""

    @abc.abstractmethod
    def get_equity(self) -> float:
        ...

    # -- protective (broker-side) stops ------------------------------------
    # Optional: a broker that can rest a real stop order server-side
    # overrides these, so a position stays protected even while this
    # process is halted, crashed, restarting or offline. The defaults are
    # no-ops (PaperBroker, LiveBrokerStub): there is no server to rest an
    # order on, and Orchestrator's own per-bar _check_stop_losses() still
    # enforces the stop locally either way.

    def place_protective_stop(self, ticker: str, close_action: Action, quantity: float,
                              stop_price: float) -> str | None:
        """Rest a stop order that closes `quantity` (`close_action` is the
        CLOSING side: SELL for a long, BUY for a short) if price trades
        through `stop_price`. Returns a broker order id, or None if no
        stop was placed (unsupported, rejected, or failed)."""
        return None

    def cancel_protective_stop(self, order_id: str, timestamp) -> tuple[str, Fill | None]:
        """Release a stop placed by place_protective_stop() before the
        position is closed some other way. Returns one of:
          ("canceled", None) -- the stop is gone; caller closes normally.
          ("filled", Fill)   -- the stop ALREADY executed at the broker;
                                the position is closed, so the caller must
                                NOT submit its own closing order (that would
                                open a position in the opposite direction).
          ("unknown", None)  -- couldn't confirm either way; the caller
                                should not close more than the broker still
                                reports holding."""
        return ("canceled", None)

    @abc.abstractmethod
    def get_position(self, ticker: str) -> Position:
        ...


class PaperBroker(Broker):
    def __init__(self, starting_cash: float = 100_000.0, slippage_bps: float = 2.0,
                 commission_bps: float = 1.0):
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []

    def get_position(self, ticker: str) -> Position:
        return self.positions.get(ticker, Position(ticker=ticker))

    def get_equity(self, mark_prices: dict[str, float] | None = None) -> float:
        equity = self.cash
        if mark_prices:
            for t, pos in self.positions.items():
                equity += pos.quantity * mark_prices.get(t, pos.avg_price)
        else:
            for pos in self.positions.values():
                equity += pos.quantity * pos.avg_price
        return equity

    def _execute(self, ticker: str, action: Action, quantity: float, ref_price: float, timestamp) -> Fill | None:
        """Shared fill/position/cash math for both `submit_order` (which
        derives `quantity` from a dollar notional) and `close_quantity`
        (which is handed `quantity` directly) -- factored out so the two
        entry points can never apply the slippage/commission/avg-price
        logic differently from each other."""
        if action == Action.HOLD or quantity <= 0:
            return None

        slip = ref_price * (self.slippage_bps / 10_000.0)
        fill_price = ref_price + slip if action == Action.BUY else ref_price - slip
        notional = quantity * fill_price
        commission = notional * (self.commission_bps / 10_000.0)

        pos = self.positions.setdefault(ticker, Position(ticker=ticker))
        signed_qty = quantity if action == Action.BUY else -quantity

        if action == Action.BUY:
            self.cash -= (quantity * fill_price + commission)
        else:
            self.cash += (quantity * fill_price - commission)

        new_qty = pos.quantity + signed_qty
        if new_qty == 0:
            pos.avg_price = 0.0
        elif pos.quantity * signed_qty >= 0:
            # Opening or adding: weighted average of old basis and this fill.
            pos.avg_price = (pos.avg_price * pos.quantity + fill_price * signed_qty) / new_qty
        elif pos.quantity * new_qty < 0:
            # Flipped through zero: the remainder was opened at this fill.
            pos.avg_price = fill_price
        # A partial reduce keeps the remaining shares' original basis, as a
        # real broker reports it -- unrealized P/L is measured against it.
        pos.quantity = new_qty

        fill = Fill(ticker=ticker, action=action, quantity=quantity, price=fill_price,
                    commission=commission, timestamp=timestamp)
        self.fills.append(fill)
        return fill

    def submit_order(self, ticker: str, action: Action, notional: float, ref_price: float, timestamp) -> Fill | None:
        if action == Action.HOLD or notional <= 0:
            return None
        slip = ref_price * (self.slippage_bps / 10_000.0)
        fill_price = ref_price + slip if action == Action.BUY else ref_price - slip
        quantity = notional / fill_price
        return self._execute(ticker, action, quantity, ref_price, timestamp)

    def close_quantity(self, ticker: str, action: Action, quantity: float, ref_price: float, timestamp) -> Fill | None:
        return self._execute(ticker, action, quantity, ref_price, timestamp)


class LiveBrokerStub(Broker):
    """Skeleton for a real broker integration. NOT functional -- wire this
    up to a vetted broker SDK (e.g. alpaca-py, ib_insync) before use, and
    only after this strategy has a long, clean paper-trading track record.

    Things a real implementation must add that PaperBroker glosses over:
    - actual order acknowledgement / partial fills / rejections
    - API rate limits and retry/backoff logic
    - reconciliation between the broker's reported positions and this
      system's internal state (treat the broker as the source of truth)
    - market hours / halted-security handling
    - authentication and secrets management (never hardcode API keys)
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        raise NotImplementedError(
            "LiveBrokerStub is a placeholder. Implement submit_order/close_quantity/"
            "get_equity/get_position against a real, tested broker SDK before using "
            "this class."
        )

    def submit_order(self, ticker: str, action: Action, notional: float, ref_price: float, timestamp) -> Fill | None:
        raise NotImplementedError

    def close_quantity(self, ticker: str, action: Action, quantity: float, ref_price: float, timestamp) -> Fill | None:
        raise NotImplementedError

    def get_equity(self) -> float:
        raise NotImplementedError

    def get_position(self, ticker: str) -> Position:
        raise NotImplementedError
