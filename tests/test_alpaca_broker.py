import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.execution.alpaca_broker import AlpacaBroker


def test_refuses_to_construct_outside_paper_mode():
    # This guard is the whole point of the class: it must be impossible to
    # end up with a live-money AlpacaBroker by accident (wrong default,
    # typo'd config, copy-pasted example). The check has to happen before
    # any network call or real `alpaca` client construction.
    with pytest.raises(ValueError, match="paper"):
        AlpacaBroker(api_key="x", secret_key="y", paper=False)


def test_constructs_in_paper_mode():
    broker = AlpacaBroker(api_key="x", secret_key="y", paper=True)
    assert broker.client is not None


class _FakeOrder:
    def __init__(self, id, status, filled_qty=None, filled_avg_price=None):
        self.id = id
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price


class _FakeClient:
    """Stands in for alpaca-py's TradingClient: submit_order returns an
    order immediately 'pending', get_order_by_id is polled until it
    reports a terminal status -- exactly the contract AlpacaBroker's
    polling loop (_submit_market_order) depends on."""
    def __init__(self, final_status, filled_qty=None, filled_avg_price=None):
        self.final_status = final_status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.submitted = []
        self._polls = 0

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        return _FakeOrder(id="order-1", status="pending_new")

    def get_order_by_id(self, order_id):
        from alpaca.trading.enums import OrderStatus
        self._polls += 1
        return _FakeOrder(id=order_id, status=self.final_status,
                           filled_qty=self.filled_qty, filled_avg_price=self.filled_avg_price)


def _broker_with_fake_client(final_status, **kw):
    broker = AlpacaBroker(api_key="x", secret_key="y", paper=True,
                           fill_poll_timeout_s=0.3, fill_poll_interval_s=0.05)
    broker.client = _FakeClient(final_status, **kw)
    return broker


class TestCloseQuantity:
    """Regression coverage for the same gap PaperBroker.close_quantity
    closes (see tests/test_broker.py and broker.py's docstring), on the
    real-Alpaca-API side: an exact-quantity closing order, sharing the
    same submit-then-poll-for-fill logic submit_order already used."""

    def test_a_filled_close_returns_a_fill_with_the_closing_action(self):
        from src.model.signals import Action
        from alpaca.trading.enums import OrderStatus
        broker = _broker_with_fake_client(OrderStatus.FILLED, filled_qty=25.0, filled_avg_price=101.5)

        fill = broker.close_quantity("AAPL", Action.SELL, quantity=25.0, ref_price=101.0, timestamp="t1")

        assert fill is not None
        assert fill.action == Action.SELL
        assert fill.quantity == pytest.approx(25.0)
        assert fill.price == pytest.approx(101.5)
        assert fill.commission == 0.0

    def test_submits_the_requested_quantity_not_a_notional_derived_one(self):
        from src.model.signals import Action
        from alpaca.trading.enums import OrderStatus, OrderSide
        broker = _broker_with_fake_client(OrderStatus.FILLED, filled_qty=12.3456, filled_avg_price=50.0)

        broker.close_quantity("AAPL", Action.BUY, quantity=12.3456, ref_price=999_999.0, timestamp="t1")

        assert len(broker.client.submitted) == 1
        submitted = broker.client.submitted[0]
        # ref_price is deliberately absurd above -- if close_quantity were
        # (incorrectly) deriving qty from notional/ref_price the way
        # submit_order does, this would submit something tiny instead of
        # the requested 12.3456.
        assert submitted.qty == pytest.approx(12.3456)
        assert submitted.side == OrderSide.BUY

    def test_a_rejected_order_returns_none(self):
        from src.model.signals import Action
        from alpaca.trading.enums import OrderStatus
        broker = _broker_with_fake_client(OrderStatus.REJECTED)

        fill = broker.close_quantity("AAPL", Action.SELL, quantity=10.0, ref_price=100.0, timestamp="t1")

        assert fill is None

    def test_a_close_that_never_confirms_within_the_poll_window_returns_none(self):
        from src.model.signals import Action
        broker = _broker_with_fake_client("pending_new")  # never reaches a terminal status

        fill = broker.close_quantity("AAPL", Action.SELL, quantity=10.0, ref_price=100.0, timestamp="t1")

        assert fill is None

    def test_hold_or_non_positive_quantity_never_submits_anything(self):
        from src.model.signals import Action
        from alpaca.trading.enums import OrderStatus
        broker = _broker_with_fake_client(OrderStatus.FILLED, filled_qty=1.0, filled_avg_price=1.0)

        assert broker.close_quantity("AAPL", Action.HOLD, quantity=10.0, ref_price=100.0, timestamp="t1") is None
        assert broker.close_quantity("AAPL", Action.SELL, quantity=0.0, ref_price=100.0, timestamp="t1") is None
        assert broker.close_quantity("AAPL", Action.SELL, quantity=-1.0, ref_price=100.0, timestamp="t1") is None
        assert broker.client.submitted == []


class _RaisingSubmitClient:
    """Stands in for a client whose submit_order() raises synchronously --
    exactly what alpaca-py does when Alpaca rejects an order outright
    (too-small notional/cost-basis, insufficient buying power, a halted
    symbol, ...) instead of accepting it and reporting REJECTED through
    get_order_by_id the way the polling loop normally handles it. Live,
    an order this small ("cost basis must be >= minimal amount of order
    1") propagated this exception all the way out of Orchestrator.run()
    and crashed the entire run_live_alpaca.py process."""
    def __init__(self, exc):
        self._exc = exc
        self.submitted = []

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        raise self._exc

    def get_order_by_id(self, order_id):
        raise AssertionError("should never poll for a fill when submit_order raised")


class _RaisingThenFilledPollClient:
    """submit_order succeeds, but get_order_by_id raises on its first call
    (a transient API/network error while polling) before succeeding on a
    later poll -- this must be treated as "not yet confirmed", not as a
    crash."""
    def __init__(self, poll_exc, final_status, filled_qty=None, filled_avg_price=None):
        self._poll_exc = poll_exc
        self.final_status = final_status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.submitted = []
        self._polls = 0

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        return _FakeOrder(id="order-1", status="pending_new")

    def get_order_by_id(self, order_id):
        self._polls += 1
        if self._polls == 1:
            raise self._poll_exc
        return _FakeOrder(id=order_id, status=self.final_status,
                           filled_qty=self.filled_qty, filled_avg_price=self.filled_avg_price)


class TestSubmitOrderSurvivesApiErrors:
    """Regression coverage for the crash found live in
    run_live_alpaca_stdout.log: an uncaught alpaca.common.exceptions.APIError
    (or any other exception) from submit_order()/get_order_by_id() must
    never propagate out of AlpacaBroker -- it must come back as "no fill"
    so the live trading loop keeps running instead of dying until the next
    scheduled restart."""

    def test_a_synchronous_rejection_from_submit_order_returns_none(self):
        from src.model.signals import Action
        from alpaca.common.exceptions import APIError
        broker = AlpacaBroker(api_key="x", secret_key="y", paper=True,
                               fill_poll_timeout_s=0.3, fill_poll_interval_s=0.05)
        exc = APIError('{"code":40310000,"message":"cost basis must be >= minimal amount of order 1"}')
        broker.client = _RaisingSubmitClient(exc)

        fill = broker.submit_order("AAPL", Action.BUY, notional=50.0, ref_price=100.0, timestamp="t1")

        assert fill is None
        assert len(broker.client.submitted) == 1

    def test_a_submit_order_exception_does_not_propagate(self):
        from src.model.signals import Action
        broker = AlpacaBroker(api_key="x", secret_key="y", paper=True,
                               fill_poll_timeout_s=0.3, fill_poll_interval_s=0.05)
        broker.client = _RaisingSubmitClient(RuntimeError("boom"))

        # Must not raise -- this is the exact failure mode that crashed
        # run_live_alpaca.py: the exception reaching Orchestrator.run().
        fill = broker.submit_order("AAPL", Action.BUY, notional=50.0, ref_price=100.0, timestamp="t1")
        assert fill is None

    def test_notional_below_alpacas_one_dollar_minimum_never_submits(self):
        from src.model.signals import Action
        broker = AlpacaBroker(api_key="x", secret_key="y", paper=True,
                               fill_poll_timeout_s=0.3, fill_poll_interval_s=0.05)
        broker.client = _RaisingSubmitClient(RuntimeError("should never be called"))

        fill = broker.submit_order("AAPL", Action.BUY, notional=0.5, ref_price=100.0, timestamp="t1")

        assert fill is None
        assert broker.client.submitted == []

    def test_a_transient_poll_error_is_retried_not_raised(self):
        from src.model.signals import Action
        from alpaca.trading.enums import OrderStatus
        broker = AlpacaBroker(api_key="x", secret_key="y", paper=True,
                               fill_poll_timeout_s=0.5, fill_poll_interval_s=0.05)
        broker.client = _RaisingThenFilledPollClient(
            ConnectionError("transient"), OrderStatus.FILLED,
            filled_qty=1.0, filled_avg_price=50.0,
        )

        fill = broker.submit_order("AAPL", Action.BUY, notional=50.0, ref_price=100.0, timestamp="t1")

        assert fill is not None
        assert fill.quantity == pytest.approx(1.0)
        assert fill.price == pytest.approx(50.0)

    def test_close_quantity_also_survives_a_raising_submit_order(self):
        from src.model.signals import Action
        broker = AlpacaBroker(api_key="x", secret_key="y", paper=True,
                               fill_poll_timeout_s=0.3, fill_poll_interval_s=0.05)
        broker.client = _RaisingSubmitClient(RuntimeError("boom"))

        fill = broker.close_quantity("AAPL", Action.SELL, quantity=10.0, ref_price=100.0, timestamp="t1")

        assert fill is None


class _FakePosition:
    def __init__(self, qty, qty_available=None):
        self.qty = qty
        self.qty_available = qty_available
        self.avg_entry_price = "100.0"


class _PositionClient(_FakeClient):
    """_FakeClient plus get_open_position, returning `position` (or
    raising, like alpaca-py does for a symbol with no open position)."""
    def __init__(self, position, **kw):
        from alpaca.trading.enums import OrderStatus
        super().__init__(OrderStatus.FILLED, **kw)
        self.position = position

    def get_open_position(self, ticker):
        if self.position is None:
            raise Exception("position does not exist")
        return self.position


def _broker_with_position(position, **kw):
    broker = AlpacaBroker(api_key="x", secret_key="y", paper=True,
                           fill_poll_timeout_s=0.3, fill_poll_interval_s=0.05)
    broker.client = _PositionClient(position, **kw)
    return broker


class TestSellNeverExceedsAvailableQty:
    """Regression for the live rejection submit_order(UNH, qty=1.5831) ->
    'insufficient qty available (available: 0.0929)': the account is one
    netted position per symbol, so a close sized from an entry's own Fill
    can ask for more shares than are still held."""

    def test_close_larger_than_available_sells_only_what_is_available(self):
        from src.model.signals import Action
        broker = _broker_with_position(_FakePosition("0.0929", "0.0929"),
                                        filled_qty=0.0929, filled_avg_price=330.0)

        fill = broker.close_quantity("UNH", Action.SELL, quantity=1.5831, ref_price=330.0, timestamp="t1")

        assert broker.client.submitted[0].qty == pytest.approx(0.0929)
        assert fill is not None and fill.quantity == pytest.approx(0.0929)

    def test_sell_entry_larger_than_available_is_clamped_too(self):
        from src.model.signals import Action
        broker = _broker_with_position(_FakePosition("0.0929", "0.0929"),
                                        filled_qty=0.0929, filled_avg_price=330.0)

        broker.submit_order("UNH", Action.SELL, notional=1.5831 * 330.0, ref_price=330.0, timestamp="t1")

        assert broker.client.submitted[0].qty == pytest.approx(0.0929)

    def test_uses_qty_available_not_qty(self):
        # Shares reserved by another still-open order count as held but
        # not available -- Alpaca rejects against qty_available.
        from src.model.signals import Action
        broker = _broker_with_position(_FakePosition("2.0", "0.5"),
                                        filled_qty=0.5, filled_avg_price=330.0)

        broker.close_quantity("UNH", Action.SELL, quantity=1.5831, ref_price=330.0, timestamp="t1")

        assert broker.client.submitted[0].qty == pytest.approx(0.5)

    def test_nothing_available_submits_nothing(self):
        from src.model.signals import Action
        broker = _broker_with_position(_FakePosition("2.0", "0"))

        fill = broker.close_quantity("UNH", Action.SELL, quantity=1.5831, ref_price=330.0, timestamp="t1")

        assert fill is None
        assert broker.client.submitted == []

    def test_sell_within_available_is_unchanged(self):
        from src.model.signals import Action
        broker = _broker_with_position(_FakePosition("5.0", "5.0"),
                                        filled_qty=1.5831, filled_avg_price=330.0)

        broker.close_quantity("UNH", Action.SELL, quantity=1.5831, ref_price=330.0, timestamp="t1")

        assert broker.client.submitted[0].qty == pytest.approx(1.5831)

    def test_no_open_position_passes_the_order_through_unchanged(self):
        from src.model.signals import Action
        broker = _broker_with_position(None, filled_qty=1.5831, filled_avg_price=330.0)

        broker.close_quantity("UNH", Action.SELL, quantity=1.5831, ref_price=330.0, timestamp="t1")

        assert broker.client.submitted[0].qty == pytest.approx(1.5831)

    def test_buys_are_never_clamped(self):
        from src.model.signals import Action
        broker = _broker_with_position(_FakePosition("0.0929", "0.0929"),
                                        filled_qty=1.5831, filled_avg_price=330.0)

        broker.close_quantity("UNH", Action.BUY, quantity=1.5831, ref_price=330.0, timestamp="t1")

        assert broker.client.submitted[0].qty == pytest.approx(1.5831)
