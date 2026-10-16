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
        except Exception as exc:
            # Alpaca answers 404 / code 40410000 when there is simply no
            # position. Anything else (timeout, 5xx, auth) means we don't
            # know, and must not be read as flat: startup reconciliation
            # would otherwise drop a saved position that is still open.
            if self._is_no_position(exc):
                return Position(ticker=ticker)
            print(f"[AlpacaBroker] get_open_position({ticker}) raised {exc!r} -- position unknown.")
            return Position(ticker=ticker, unknown=True)
        return Position(ticker=ticker, quantity=float(pos.qty), avg_price=float(pos.avg_entry_price))

    @staticmethod
    def _is_no_position(exc: Exception) -> bool:
        for attr in ("status_code", "code"):
            try:
                if getattr(exc, attr, None) in (404, 40410000):
                    return True
            except Exception:  # noqa: BLE001 -- APIError.code parses JSON
                pass
        return "position does not exist" in str(exc).lower()

    def _clamp_sell_qty(self, ticker: str, qty: float) -> float | None:
        """Cap a SELL at the long shares Alpaca says are actually available.

        Observed live: submit_order(UNH, qty=1.5831) was rejected with
        "insufficient qty available (available: 0.0929)". Orchestrator
        sizes each closing SELL from the entry's own Fill
        (`p["filled_quantity"]`), but the Alpaca account is a single netted position per
        symbol: an earlier SELL signal (or another entry's close) has
        often already sold most of those shares, so the requested qty
        no longer exists. Alpaca can't flip a long to a short in one
        order, and fractional shares can't be shorted at all, so the
        whole order was rejected and the shares that WERE still held
        stayed open. Selling `min(qty, qty_available)` closes what is
        really there; the Fill reports the quantity that actually traded.

        `qty_available` (not `qty`) is what Alpaca checks against -- it
        excludes shares already held by other open orders, e.g. a
        previous close that timed out in our poll window but is still
        working at Alpaca.

        Only applied to `close_quantity`, never to a SELL entry from
        `submit_order`: Orchestrator records a filled SELL entry as a new
        short and later BUYs to close it, so silently turning an
        oversized SELL entry into a partial liquidation of an existing
        long would make that later BUY open an unintended long.

        Returns None (submit nothing) when a long position exists but
        none of it is available. With no long position (lookup fails or
        qty <= 0) the order is passed through unchanged -- same behavior
        as before this check existed."""
        try:
            pos = self.client.get_open_position(ticker)
            held = float(pos.qty)
            raw_available = getattr(pos, "qty_available", None)
            available = float(raw_available if raw_available is not None else pos.qty)
        except Exception:
            return qty
        if held <= 0:
            return qty
        if available <= 0:
            print(f"[AlpacaBroker] SELL {ticker} qty={qty} skipped -- position holds {held} "
                  f"but none is available (held by other open orders).")
            return None
        if qty > available:
            print(f"[AlpacaBroker] SELL {ticker} qty={qty} exceeds available {available} "
                  f"-- selling {available} instead.")
            return available
        return qty

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
        try:
            order = self.client.submit_order(order_data=order_req)
        except Exception as exc:
            # Alpaca rejects some orders synchronously, by raising out of
            # submit_order(), instead of accepting them and reporting
            # REJECTED through get_order_by_id the way the polling loop
            # below handles it (too-small notional/cost-basis,
            # insufficient buying power, a halted symbol, ...). Observed
            # live: a single order this small ("cost basis must be >=
            # minimal amount of order 1") took down the entire
            # run_live_alpaca.py process, since nothing here caught it --
            # it propagated all the way out of Orchestrator.run() and
            # killed the live trading loop until the next scheduled
            # premarket_check.py restart found it not running. Same
            # treatment as a REJECTED order a few lines down: no Fill,
            # loudly logged, loop keeps running.
            print(f"[AlpacaBroker] submit_order({ticker}, qty={qty}) raised {exc!r} -- treating as no fill.")
            return None

        deadline = time.monotonic() + self.fill_poll_timeout_s
        while time.monotonic() < deadline:
            try:
                order = self.client.get_order_by_id(order.id)
            except Exception as exc:
                # Same reasoning as above -- a transient API/network error
                # while polling for a fill must not crash the live loop
                # either. Treated as "not confirmed yet" for this poll;
                # the outer while loop retries until fill_poll_timeout_s.
                print(f"[AlpacaBroker] get_order_by_id({order.id}) raised {exc!r} while polling "
                      f"for a fill -- treating as not yet confirmed.")
                time.sleep(self.fill_poll_interval_s)
                continue
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

        if notional < 1.0:
            # Alpaca's own minimum for a notional-based order is $1 --
            # exactly the condition behind the "cost basis must be >=
            # minimal amount of order 1" rejection seen live. Skip
            # submitting something it will just reject rather than
            # relying solely on _submit_market_order's exception handling
            # to catch it after the fact.
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

        if action == Action.SELL:
            qty = self._clamp_sell_qty(ticker, qty)
            if qty is None:
                return None

        return self._submit_market_order(ticker, action, qty, timestamp)

    # -- protective (broker-side) stops ------------------------------------

    def place_protective_stop(self, ticker: str, close_action: Action, quantity: float,
                              stop_price: float) -> str | None:
        """Rest a real stop order at Alpaca for `quantity` shares, so the
        position is protected even if this process halts, crashes or is
        offline -- the per-bar local stop check only runs while Python is
        running and receiving bars.

        TimeInForce.DAY, not GTC: Alpaca only accepts DAY for fractional
        quantities, and this system's quantities are fractional
        (notional / price). A DAY stop expires at the close, so
        Orchestrator re-places stops for every tracked position at
        startup (_reconcile_positions), which premarket_check.py's daily
        restart already guarantees happens each morning.

        Never raises: a rejected stop (e.g. stop_price already through
        the market) returns None and is logged; the local per-bar stop
        check is still the backstop."""
        if close_action == Action.HOLD or quantity <= 0 or not (stop_price > 0):
            return None
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest

        req = StopOrderRequest(
            symbol=ticker, qty=round(quantity, 4),
            side=OrderSide.SELL if close_action == Action.SELL else OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            stop_price=round(stop_price, 2),
        )
        try:
            order = self.client.submit_order(order_data=req)
        except Exception as exc:
            print(f"[AlpacaBroker] protective stop for {ticker} (qty={quantity}, "
                  f"stop={stop_price:.2f}) rejected: {exc!r} -- local stop check still applies.")
            return None
        return str(order.id)

    @staticmethod
    def _is_filled(status) -> bool:
        from alpaca.trading.enums import OrderStatus
        # PARTIALLY_FILLED is deliberately not "filled": the remainder is
        # still resting and gets canceled below like any open order.
        return status == OrderStatus.FILLED or str(status).lower() in ("filled", "orderstatus.filled")

    @staticmethod
    def _is_dead(status) -> bool:
        from alpaca.trading.enums import OrderStatus
        dead = (OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED)
        return status in dead or str(status).lower().split(".")[-1] in ("canceled", "expired", "rejected")

    @staticmethod
    def _partial(order) -> bool:
        try:
            return float(getattr(order, "filled_qty", 0) or 0) > 0
        except (TypeError, ValueError):
            return False

    def _stop_fill(self, order, timestamp) -> Fill:
        side = getattr(order, "side", None)
        action = Action.BUY if str(side).lower().endswith("buy") else Action.SELL
        return Fill(ticker=str(getattr(order, "symbol", "")), action=action,
                    quantity=float(order.filled_qty), price=float(order.filled_avg_price),
                    commission=0.0, timestamp=timestamp)

    def cancel_protective_stop(self, order_id: str, timestamp) -> tuple[str, Fill | None]:
        """See Broker.cancel_protective_stop. Checks first whether the stop
        already executed (then the position is gone and the caller must
        not sell again), otherwise cancels it and polls until Alpaca
        confirms the cancel -- the shares it was holding aren't released
        for a closing market order until then."""
        try:
            order = self.client.get_order_by_id(order_id)
        except Exception as exc:
            print(f"[AlpacaBroker] get_order_by_id({order_id}) raised {exc!r} -- stop status unknown.")
            return ("unknown", None)
        if self._is_filled(order.status):
            return ("filled", self._stop_fill(order, timestamp))
        if self._is_dead(order.status):
            return ("unknown" if self._partial(order) else "canceled", None)
        try:
            self.client.cancel_order_by_id(order_id)
        except Exception as exc:
            # Commonly "order is already filled" racing us -- re-check below.
            print(f"[AlpacaBroker] cancel_order_by_id({order_id}) raised {exc!r} -- re-checking.")
        deadline = time.monotonic() + self.fill_poll_timeout_s
        while True:
            try:
                order = self.client.get_order_by_id(order_id)
            except Exception:
                order = None
            if order is not None:
                if self._is_filled(order.status):
                    return ("filled", self._stop_fill(order, timestamp))
                if self._is_dead(order.status):
                    # A stop that partially filled before the cancel landed
                    # already closed SOME shares: let the caller clamp its
                    # close to what the broker still holds.
                    return ("unknown" if self._partial(order) else "canceled", None)
            if time.monotonic() >= deadline:
                return ("unknown", None)
            time.sleep(self.fill_poll_interval_s)
