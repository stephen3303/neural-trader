"""
Paper (and, in principle, live) order execution via Alpaca.

This is the real implementation `LiveBrokerStub` in broker.py deliberately
withheld. It satisfies the same `Broker` interface as `PaperBroker`, so
`Orchestrator` doesn't know or care that fills are now coming from a real
brokerage API instead of a slippage model.

Safety: the constructor refuses to run unless `paper=True`. That's not a
suggestion -- flip it only once you've read the README's "before you even
think about live trading" section and mean it. There is no flag that
silently makes this safe; you'd have to deliberately edit this file.

Fill confirmation: Alpaca's `submit_order` returns immediately with the
order in `accepted`/`pending_new` status, not yet filled. This polls
`get_order_by_id` for a short window waiting for a terminal `filled`
status, since the rest of the system (ReplayBuffer labels, the decision
log, risk state) expects `submit_order` to return a `Fill` synchronously,
the same contract `PaperBroker` uses. Market orders during regular trading
hours typically fill within a second or two; if the poll window elapses
first, this returns None (treated as "no fill yet") rather than guessing --
the order may still be open at Alpaca, so check there if that happens
often. `close_quantity` shares this exact same polling behavior (see
`_submit_market_order` below) -- a stop-loss/maturity close that doesn't
confirm within the poll window also returns None rather than guessing,
which Orchestrator logs (DecisionLogger.log_close_order) instead of
silently assuming it worked.

Commission: Alpaca is commission-free for US equities, so fills always
carry commission=0.0 here (unlike PaperBroker, which applies a configurable
bps model for a more conservative/general-purpose paper estimate).
"""

from __future__ import annotations

import time

from src.model.signals import Action
from .broker import Broker, Fill, Position


class AlpacaBroker(Broker):
    def __init__(self, api_key: str, secret_key: str, paper: bool = True,
                 fill_poll_timeout_s: float = 10.0, fill_poll_interval_s: float = 0.5):
        if not paper:
            raise ValueError(
                "AlpacaBroker refuses to construct with paper=False. This scaffold's "
                "risk controls have not been validated for live-money trading -- see "
                "the README's 'Before you even think about live trading' section. "
                "If you've genuinely decided to go live, that is a deliberate code "
                "change to make here, not a config flag to flip."
            )
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(api_key, secret_key, paper=True)
        self.fill_poll_timeout_s = fill_poll_timeout_s
        self.fill_poll_interval_s = fill_poll_interval_s

    def get_equity(self, mark_prices: dict[str, float] | None = None) -> float:
        # Alpaca's account object already marks open positions to market
        # server-side, so mark_prices (used by PaperBroker, which has no
        # such server to ask) is accepted for interface compatibility and
        # otherwise ignored here.
        account = self.client.get_account()
        return float(account.equity)

    def get_position(self, ticker: str) -> Position:
        try:
            pos = self.client.get_open_position(ticker)
        except Exception:
            return Position(ticker=ticker)
        return Position(ticker=ticker, quantity=float(pos.qty), avg_price=float(pos.avg_entry_price))

    def _submit_market_order(self, ticker: str, action: Action, qty: float, timestamp) -> Fill | None:
        """Shared order-submission + fill-polling logic for both
        `submit_order` (qty derived from a dollar notional) and
        `close_quantity` (qty given directly) -- see each caller's own
        qty/validity checks; this assumes `qty` is already a valid,
        positive, non-HOLD quantity."""
        from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order_req = MarketOrderRequest(
            symbol=ticker, qty=qty,
            side=OrderSide.BUY if action == Action.BUY else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(order_data=order_req)

        deadline = time.monotonic() + self.fill_poll_timeout_s
        while time.monotonic() < deadline:
            order = self.client.get_order_by_id(order.id)
            status = getattr(order, "status", None)
            if status == OrderStatus.FILLED or str(status).lower().endswith("filled"):
                return Fill(
                    ticker=ticker, action=action,
                    quantity=float(order.filled_qty), price=float(order.filled_avg_price),
                    commission=0.0, timestamp=timestamp,
                )
            if status in (OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED):
                return None
            time.sleep(self.fill_poll_interval_s)

        # Still open at Alpaca when we stopped polling -- not a Fill yet.
        # It isn't canceled here: on the next bar, risk sizing just treats
        # this as "no position change happened," which is conservative.
        return None

    def submit_order(self, ticker: str, action: Action, notional: float, ref_price: float, timestamp) -> Fill | None:
        if action == Action.HOLD or notional <= 0 or ref_price <= 0:
            return None

        qty = round(notional / ref_price, 4)
        if qty <= 0:
            return None

        return self._submit_market_order(ticker, action, qty, timestamp)

    def close_quantity(self, ticker: str, action: Action, quantity: float, ref_price: float, timestamp) -> Fill | None:
        # ref_price is accepted for interface symmetry with PaperBroker
        # (which needs it to compute slippage) and otherwise unused here:
        # a real market order fills at whatever Alpaca's actual market
        # price is, not a pre-computed reference price.
        if action == Action.HOLD or quantity <= 0:
            return None

        qty = round(quantity, 4)
        if qty <= 0:
            return None

        return self._submit_market_order(ticker, action, qty, timestamp)
