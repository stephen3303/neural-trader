"""Direct unit tests for PaperBroker -- previously only ever exercised
indirectly through Orchestrator tests, which meant submit_order's own
cash/position/commission math had no isolated regression coverage.
Added together with close_quantity (see broker.py's docstring on it and
the README): the fix for a real bug where closing out a prediction's
exit never submitted any order to the broker at all."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.execution.broker import Fill, PaperBroker, Position
from src.model.signals import Action


class TestSubmitOrder:
    def test_a_buy_increases_position_and_decreases_cash(self):
        broker = PaperBroker(starting_cash=100_000.0, slippage_bps=0.0, commission_bps=0.0)
        fill = broker.submit_order("T", Action.BUY, notional=10_000.0, ref_price=100.0, timestamp="t0")
        assert fill is not None
        assert fill.quantity == pytest.approx(100.0)
        assert broker.get_position("T").quantity == pytest.approx(100.0)
        assert broker.cash == pytest.approx(90_000.0)

    def test_a_sell_opens_a_short_and_increases_cash(self):
        broker = PaperBroker(starting_cash=100_000.0, slippage_bps=0.0, commission_bps=0.0)
        fill = broker.submit_order("T", Action.SELL, notional=10_000.0, ref_price=100.0, timestamp="t0")
        assert fill is not None
        assert broker.get_position("T").quantity == pytest.approx(-100.0)
        assert broker.cash == pytest.approx(110_000.0)

    def test_hold_or_non_positive_notional_is_rejected(self):
        broker = PaperBroker()
        assert broker.submit_order("T", Action.HOLD, notional=10_000.0, ref_price=100.0, timestamp="t0") is None
        assert broker.submit_order("T", Action.BUY, notional=0.0, ref_price=100.0, timestamp="t0") is None
        assert broker.submit_order("T", Action.BUY, notional=-5.0, ref_price=100.0, timestamp="t0") is None

    def test_slippage_makes_a_buy_fill_above_and_a_sell_fill_below_ref_price(self):
        broker = PaperBroker(slippage_bps=100.0, commission_bps=0.0)  # 1% slippage, exaggerated for a clear assertion
        buy = broker.submit_order("T", Action.BUY, notional=10_000.0, ref_price=100.0, timestamp="t0")
        assert buy.price == pytest.approx(101.0)
        sell = broker.submit_order("T2", Action.SELL, notional=10_000.0, ref_price=100.0, timestamp="t0")
        assert sell.price == pytest.approx(99.0)

    def test_commission_is_deducted_on_top_of_notional(self):
        broker = PaperBroker(starting_cash=100_000.0, slippage_bps=0.0, commission_bps=10.0)  # 0.1%
        broker.submit_order("T", Action.BUY, notional=10_000.0, ref_price=100.0, timestamp="t0")
        assert broker.cash == pytest.approx(100_000.0 - 10_000.0 - 10.0)


class TestCloseQuantity:
    """The fix: an exact-share-count counterpart to submit_order's
    dollar-notional sizing, so Orchestrator can flatten precisely the
    quantity a prior entry's Fill reported, instead of re-deriving a
    dollar amount that would (after slippage) only approximately net
    back out to the right number of shares."""

    def test_closing_a_long_position_sells_the_exact_quantity_flat(self):
        broker = PaperBroker(slippage_bps=0.0, commission_bps=0.0)
        broker.positions["T"] = Position(ticker="T", quantity=100.0, avg_price=100.0)
        fill = broker.close_quantity("T", Action.SELL, quantity=100.0, ref_price=110.0, timestamp="t1")
        assert fill is not None
        assert fill.quantity == pytest.approx(100.0)
        assert broker.get_position("T").quantity == pytest.approx(0.0)

    def test_closing_a_short_position_buys_the_exact_quantity_flat(self):
        broker = PaperBroker(slippage_bps=0.0, commission_bps=0.0)
        broker.positions["T"] = Position(ticker="T", quantity=-40.0, avg_price=100.0)
        fill = broker.close_quantity("T", Action.BUY, quantity=40.0, ref_price=90.0, timestamp="t1")
        assert fill is not None
        assert broker.get_position("T").quantity == pytest.approx(0.0)

    def test_closing_less_than_the_full_position_leaves_a_residual(self):
        broker = PaperBroker(slippage_bps=0.0, commission_bps=0.0)
        broker.positions["T"] = Position(ticker="T", quantity=100.0, avg_price=100.0)
        broker.close_quantity("T", Action.SELL, quantity=30.0, ref_price=110.0, timestamp="t1")
        assert broker.get_position("T").quantity == pytest.approx(70.0)

    def test_closing_updates_cash_using_the_same_slippage_and_commission_model_as_submit_order(self):
        broker = PaperBroker(starting_cash=100_000.0, slippage_bps=100.0, commission_bps=10.0)
        broker.positions["T"] = Position(ticker="T", quantity=100.0, avg_price=100.0)
        cash_before = broker.cash
        fill = broker.close_quantity("T", Action.SELL, quantity=100.0, ref_price=110.0, timestamp="t1")
        # SELL fill price = ref_price - slippage = 110 * (1 - 0.01) = 108.9
        assert fill.price == pytest.approx(108.9)
        commission = (100.0 * 108.9) * (10.0 / 10_000.0)
        assert broker.cash == pytest.approx(cash_before + 100.0 * 108.9 - commission)

    def test_hold_or_non_positive_quantity_is_rejected(self):
        broker = PaperBroker()
        broker.positions["T"] = Position(ticker="T", quantity=100.0, avg_price=100.0)
        assert broker.close_quantity("T", Action.HOLD, quantity=100.0, ref_price=110.0, timestamp="t1") is None
        assert broker.close_quantity("T", Action.SELL, quantity=0.0, ref_price=110.0, timestamp="t1") is None
        assert broker.close_quantity("T", Action.SELL, quantity=-5.0, ref_price=110.0, timestamp="t1") is None
        # Rejected -- position must be untouched.
        assert broker.get_position("T").quantity == pytest.approx(100.0)

    def test_closing_a_ticker_with_no_existing_position_still_executes(self):
        """close_quantity doesn't require a pre-existing position --
        Orchestrator always calls it with a real prior entry in mind, but
        the broker itself has no way to enforce that, and shouldn't
        silently no-op if the caller's bookkeeping and the broker's own
        position state have ever drifted apart (that mismatch is exactly
        the kind of thing a real integration's reconciliation step --
        see LiveBrokerStub's docstring -- exists to catch)."""
        broker = PaperBroker(slippage_bps=0.0, commission_bps=0.0)
        fill = broker.close_quantity("T", Action.SELL, quantity=10.0, ref_price=100.0, timestamp="t1")
        assert fill is not None
        assert broker.get_position("T").quantity == pytest.approx(-10.0)

    def test_records_the_fill_with_the_closing_action_not_the_original_entrys(self):
        broker = PaperBroker(slippage_bps=0.0, commission_bps=0.0)
        broker.positions["T"] = Position(ticker="T", quantity=100.0, avg_price=100.0)
        fill = broker.close_quantity("T", Action.SELL, quantity=100.0, ref_price=110.0, timestamp="t1")
        assert fill.action == Action.SELL
        assert fill in broker.fills
