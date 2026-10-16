"""Open positions survive restarts and get broker-side protective stops.

Regression coverage for a live incident: after a restart, Orchestrator
started with an empty `_pending`, so every position already held at the
broker (about half the account) had no stop-loss and no time exit, and --
with the kill switch engaged -- no way to be sold at all.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.execution.broker import Fill, PaperBroker, Position
from src.model.signals import Action
from tests.test_orchestrator import _FixedCloseBar, _OHLCBar, _enqueue_pending, _make_orchestrator


class _StopBroker(PaperBroker):
    """PaperBroker plus a scriptable broker-side stop, like AlpacaBroker."""

    def __init__(self, cancel_result=("canceled", None)):
        super().__init__()
        self.placed = []
        self.canceled = []
        self.closes = []
        self.cancel_result = cancel_result

    def place_protective_stop(self, ticker, close_action, quantity, stop_price):
        self.placed.append((ticker, close_action, quantity, stop_price))
        return f"stop-{len(self.placed)}"

    def cancel_protective_stop(self, order_id, timestamp):
        self.canceled.append(order_id)
        return self.cancel_result

    def close_quantity(self, ticker, action, quantity, ref_price, timestamp):
        self.closes.append((ticker, action, quantity))
        return super().close_quantity(ticker, action, quantity, ref_price, timestamp)


def _orch(tmp_path, broker=None):
    orch = _make_orchestrator(tmp_path)
    if broker is not None:
        orch.broker = broker
    orch.positions_state_path = tmp_path / "checkpoints" / "open_positions.json"
    return orch


def _hold(broker, ticker, qty, avg):
    broker.positions[ticker] = Position(ticker=ticker, quantity=qty, avg_price=avg)


def _events(tmp_path, kind):
    out = []
    for line in (tmp_path / "decisions.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["type"] == kind:
            out.append(e)
    return out


class TestReconcileOnStartup:
    def test_an_orphaned_broker_position_is_adopted_with_a_stop_and_time_exit(self, tmp_path):
        orch = _orch(tmp_path)
        t = orch.tickers[0]
        _hold(orch.broker, t, 10.0, 100.0)

        orch._reconcile_positions()

        assert len(orch._pending[t]) == 1
        p = orch._pending[t][0]
        assert p["action"] == 2 and p["filled_quantity"] == pytest.approx(10.0)
        assert p["entry_price"] == pytest.approx(100.0)
        assert p["stop_loss_pct"] == pytest.approx(orch.risk_manager.cfg.min_stop_loss_pct)
        assert p["mature_at_count"] == orch._bar_count[t] + orch.label_cfg.horizon
        assert _events(tmp_path, "position_reconciled")[0]["source"] == "adopted"
        saved = json.loads(orch.positions_state_path.read_text())
        assert saved[t][0]["filled_quantity"] == pytest.approx(10.0)

    def test_an_adopted_position_is_stopped_out_and_flattened_even_while_halted(self, tmp_path):
        orch = _orch(tmp_path)
        t = orch.tickers[0]
        _hold(orch.broker, t, 10.0, 100.0)
        orch.risk_manager.trip_kill_switch("test halt")
        orch._reconcile_positions()

        # 1.5% min stop -> 98.5; low of 98 breaches it.
        orch._check_stop_losses(t, _OHLCBar(high=100.0, low=98.0, close=99.0))

        assert len(orch._pending[t]) == 0
        assert orch.broker.get_position(t).quantity == pytest.approx(0.0)
        # Not a model prediction: must not move the drift monitor's hit-rate
        # or become a training sample.
        assert orch.drift_monitor.snapshot()["n_outcomes"] == 0
        assert len(orch.buffer) == 0
        assert json.loads(orch.positions_state_path.read_text()) == {}

    def test_an_adopted_position_closes_at_its_time_exit(self, tmp_path):
        orch = _orch(tmp_path)
        t = orch.tickers[0]
        _hold(orch.broker, t, 10.0, 100.0)
        orch._reconcile_positions()

        orch._bar_count[t] += orch.label_cfg.horizon
        orch._resolve_matured(t, _FixedCloseBar(close=100.5))

        assert len(orch._pending[t]) == 0
        assert orch.broker.get_position(t).quantity == pytest.approx(0.0)
        assert orch.trade_log[0]["pnl_pct_of_equity"] > 0

    def test_saved_entries_resume_with_their_own_stop_and_remaining_bars(self, tmp_path):
        first = _orch(tmp_path)
        t = first.tickers[0]
        _enqueue_pending(first, t, action=2, size_pct_equity=1.0, entry_price=100.0,
                          mature_at_count=7, filled_quantity=10.0, stop_loss_pct=4.0)
        first._save_positions()

        second = _orch(tmp_path)
        _hold(second.broker, t, 10.0, 100.2)
        second._bar_count[t] = 50
        second._reconcile_positions()

        assert len(second._pending[t]) == 1
        p = second._pending[t][0]
        assert p["decision_id"] == "d-test"
        assert p["stop_loss_pct"] == pytest.approx(4.0)
        assert p["entry_price"] == pytest.approx(100.0)
        assert p["mature_at_count"] == 57
        assert _events(tmp_path, "position_reconciled")[0]["source"] == "saved"

    def test_untracked_remainder_beyond_saved_entries_is_adopted(self, tmp_path):
        first = _orch(tmp_path)
        t = first.tickers[0]
        _enqueue_pending(first, t, action=2, size_pct_equity=1.0, filled_quantity=4.0, mature_at_count=5)
        first._save_positions()

        second = _orch(tmp_path)
        _hold(second.broker, t, 10.0, 100.0)
        second._reconcile_positions()

        qtys = sorted(p["filled_quantity"] for p in second._pending[t])
        assert qtys == pytest.approx([4.0, 6.0])

    def test_saved_entries_are_dropped_when_the_broker_is_flat(self, tmp_path):
        first = _orch(tmp_path)
        t = first.tickers[0]
        _enqueue_pending(first, t, action=2, size_pct_equity=1.0, filled_quantity=10.0, mature_at_count=5)
        first._save_positions()

        second = _orch(tmp_path)
        second._reconcile_positions()

        assert len(second._pending[t]) == 0
        assert json.loads(second.positions_state_path.read_text()) == {}

    def test_saved_entries_are_kept_when_the_broker_lookup_fails(self, tmp_path):
        first = _orch(tmp_path)
        t = first.tickers[0]
        _enqueue_pending(first, t, action=2, size_pct_equity=1.0, entry_price=100.0,
                          filled_quantity=10.0, mature_at_count=5, stop_loss_pct=4.0)
        first._save_positions()

        broker = _StopBroker()
        broker.get_position = lambda ticker: Position(ticker=ticker, unknown=True)
        second = _orch(tmp_path, broker)
        second._reconcile_positions()

        assert len(second._pending[t]) == 1, "an API error is not proof the position is flat"
        p = second._pending[t][0]
        assert p["filled_quantity"] == pytest.approx(10.0) and p["entry_price"] == pytest.approx(100.0)
        assert p["stop_order_id"] is not None, "the old stop was canceled, so a fresh one must be placed"
        assert t in json.loads(second.positions_state_path.read_text())

    def test_stale_saved_entries_claiming_more_than_held_are_replaced(self, tmp_path):
        first = _orch(tmp_path)
        t = first.tickers[0]
        _enqueue_pending(first, t, action=2, size_pct_equity=1.0, filled_quantity=20.0, mature_at_count=5)
        first._save_positions()

        second = _orch(tmp_path)
        _hold(second.broker, t, 10.0, 100.0)
        second._reconcile_positions()

        assert len(second._pending[t]) == 1
        assert second._pending[t][0]["filled_quantity"] == pytest.approx(10.0)
        assert second._pending[t][0]["decision_id"].startswith("adopted-")

    def test_a_short_position_is_adopted_as_a_short(self, tmp_path):
        orch = _orch(tmp_path)
        t = orch.tickers[0]
        _hold(orch.broker, t, -5.0, 100.0)
        orch._reconcile_positions()
        p = orch._pending[t][0]
        assert p["action"] == 0 and p["filled_quantity"] == pytest.approx(5.0)

    def test_no_persistence_path_means_no_file(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        orch._reconcile_positions()
        assert not (tmp_path / "checkpoints" / "open_positions.json").exists()


class TestProtectiveStops:
    def test_reconcile_places_a_broker_stop_at_the_local_stop_level(self, tmp_path):
        broker = _StopBroker()
        orch = _orch(tmp_path, broker)
        t = orch.tickers[0]
        _hold(broker, t, 10.0, 100.0)
        orch._reconcile_positions()

        assert len(broker.placed) == 1
        ticker, action, qty, stop = broker.placed[0]
        assert action == Action.SELL and qty == pytest.approx(10.0)
        assert stop == pytest.approx(100.0 * (1 - orch.risk_manager.cfg.min_stop_loss_pct / 100))
        assert orch._pending[t][0]["stop_order_id"] == "stop-1"

    def test_reconcile_cancels_the_previous_runs_stops_first(self, tmp_path):
        first = _orch(tmp_path, _StopBroker())
        t = first.tickers[0]
        _enqueue_pending(first, t, action=2, size_pct_equity=1.0, filled_quantity=10.0, mature_at_count=5)
        first._pending[t][0]["stop_order_id"] = "old-stop"
        first._save_positions()

        broker = _StopBroker()
        second = _orch(tmp_path, broker)
        _hold(broker, t, 10.0, 100.0)
        second._reconcile_positions()

        assert broker.canceled == ["old-stop"]
        assert len(broker.placed) == 1

    def test_closing_cancels_the_broker_stop_then_sells(self, tmp_path):
        broker = _StopBroker(cancel_result=("canceled", None))
        orch = _orch(tmp_path, broker)
        t = orch.tickers[0]
        _hold(broker, t, 10.0, 100.0)
        orch._reconcile_positions()

        orch._bar_count[t] += orch.label_cfg.horizon
        orch._resolve_matured(t, _FixedCloseBar(close=101.0))

        assert broker.canceled == ["stop-1"]
        assert broker.closes == [(t, Action.SELL, pytest.approx(10.0))]

    def test_a_stop_that_already_filled_at_the_broker_is_not_sold_again(self, tmp_path):
        stop_fill = Fill(ticker="T1", action=Action.SELL, quantity=10.0, price=98.4,
                         commission=0.0, timestamp="t")
        broker = _StopBroker(cancel_result=("filled", stop_fill))
        orch = _orch(tmp_path, broker)
        t = orch.tickers[0]
        _hold(broker, t, 10.0, 100.0)
        orch._reconcile_positions()

        orch._check_stop_losses(t, _OHLCBar(high=100.0, low=98.0, close=98.2))

        assert broker.closes == [], "a second sell would open a short"
        assert len(orch.trade_log) == 1
        # Booked at the broker's real stop fill, not the theoretical level.
        assert orch.trade_log[0]["pnl_pct_of_equity"] == pytest.approx(
            orch.trade_log[0]["size_pct_equity"] * (98.4 - 100.0) / 100.0)
        assert any(e["event"] == "filled" for e in _events(tmp_path, "protective_stop"))

    def test_unknown_stop_status_never_closes_more_than_the_broker_holds(self, tmp_path):
        broker = _StopBroker(cancel_result=("unknown", None))
        orch = _orch(tmp_path, broker)
        t = orch.tickers[0]
        _enqueue_pending(orch, t, action=2, size_pct_equity=1.0, filled_quantity=10.0, mature_at_count=0)
        orch._pending[t][0]["stop_order_id"] = "stop-x"
        _hold(broker, t, 4.0, 100.0)  # the stop partly executed: only 4 left

        orch._resolve_matured(t, _FixedCloseBar(close=100.0))

        assert broker.closes == [(t, Action.SELL, pytest.approx(4.0))]
        assert broker.get_position(t).quantity == pytest.approx(0.0)

    def test_new_fills_in_the_live_loop_get_a_broker_stop(self, tmp_path):
        broker = _StopBroker()
        orch = _orch(tmp_path, broker)

        def always_size(signal, vol):
            return {"ticker": signal.ticker, "action": signal.action, "size_pct_equity": 1.0,
                    "stop_loss_pct": 2.0, "reason": "sized", "model_version": signal.model_version}
        orch.risk_manager.size_order = always_size
        orch.run(max_bars=orch.warmup_bars + 3)

        assert broker.placed, "every filled entry should rest a broker-side stop"
        assert all(a in (Action.SELL, Action.BUY) for _, a, _, _ in broker.placed)
        saved = json.loads(orch.positions_state_path.read_text())
        assert saved and all(r["stop_order_id"] for rows in saved.values() for r in rows)


class TestStopsLiftedAroundNewOrders:
    def test_resting_stops_are_lifted_and_replaced_around_a_new_order(self, tmp_path):
        broker = _StopBroker()
        orch = _orch(tmp_path, broker)
        t = orch.tickers[0]
        _hold(broker, t, 10.0, 100.0)
        orch._reconcile_positions()
        assert orch._pending[t][0]["stop_order_id"] == "stop-1"

        lifted = orch._lift_protective_stops(t, "t")
        assert broker.canceled == ["stop-1"] and orch._pending[t][0]["stop_order_id"] is None
        orch._restore_protective_stops(t, lifted)
        assert orch._pending[t][0]["stop_order_id"] == "stop-2"

    def test_a_stop_found_filled_while_lifting_resolves_that_position(self, tmp_path):
        stop_fill = Fill(ticker="T1", action=Action.SELL, quantity=10.0, price=98.0,
                         commission=0.0, timestamp="t")
        broker = _StopBroker(cancel_result=("filled", stop_fill))
        orch = _orch(tmp_path, broker)
        t = orch.tickers[0]
        _hold(broker, t, 10.0, 100.0)
        orch._reconcile_positions()

        lifted = orch._lift_protective_stops(t, "t")

        assert lifted == [] and len(orch._pending[t]) == 0
        assert broker.closes == [], "already flat at the broker -- must not sell again"
        assert len(orch.trade_log) == 1


class TestWithMainExitRules:
    def test_an_unconfirmed_cancel_keeps_the_stop_and_places_no_second_one(self, tmp_path):
        broker = _StopBroker()
        orch = _orch(tmp_path, broker)
        t = orch.tickers[0]
        _hold(broker, t, 10.0, 100.0)
        orch._reconcile_positions()
        broker.cancel_result = ("unknown", None)

        lifted = orch._lift_protective_stops(t, "t")
        orch._restore_protective_stops(t, lifted)

        assert lifted == []
        assert len(broker.placed) == 1, "the original stop may still be resting -- no duplicate"
        assert orch._pending[t][0]["stop_order_id"] == "stop-1"

    def test_an_early_take_profit_releases_the_stop_and_keeps_the_label_pending(self, tmp_path):
        broker = _StopBroker(cancel_result=("canceled", None))
        orch = _orch(tmp_path, broker)
        orch.risk_manager.cfg.fixed_take_profit_mult = 1.0
        t = orch.tickers[0]
        _enqueue_pending(orch, t, action=2, size_pct_equity=1.0, entry_price=100.0,
                          mature_at_count=15, filled_quantity=10.0, stop_loss_pct=2.0)
        _hold(broker, t, 10.0, 100.0)
        orch._place_protective_stop(t, orch._pending[t][0])

        # 1R take profit at 102.
        orch._check_stop_losses(t, _OHLCBar(high=102.5, low=100.0, close=102.0))

        assert broker.canceled == ["stop-1"]
        assert broker.closes == [(t, Action.SELL, pytest.approx(10.0))]
        assert len(orch._pending[t]) == 1, "label is still recorded at the horizon"
        assert json.loads(orch.positions_state_path.read_text()) == {}

    def test_max_hold_bars_counts_from_the_original_entry_after_a_restart(self, tmp_path):
        first = _orch(tmp_path)
        t = first.tickers[0]
        _enqueue_pending(first, t, action=2, size_pct_equity=1.0, mature_at_count=15, filled_quantity=10.0)
        first._pending[t][0]["entry_count"] = 0
        first._bar_count[t] = 4
        first._save_positions()

        second = _orch(tmp_path)
        second.risk_manager.cfg.max_hold_bars = 6
        _hold(second.broker, t, 10.0, 100.0)
        second._bar_count[t] = 50
        second._reconcile_positions()

        second._bar_count[t] = 51
        second._resolve_matured(t, _FixedCloseBar(close=100.0))
        assert len(second._pending[t]) == 1
        second._bar_count[t] = 52  # 4 bars before the restart + 2 after = 6
        second._resolve_matured(t, _FixedCloseBar(close=100.0))
        assert len(second._pending[t]) == 0
        assert second.broker.get_position(t).quantity == pytest.approx(0.0)


class TestAlpacaProtectiveStops:
    class _Order:
        def __init__(self, id, status, filled_qty=0, filled_avg_price=None, side="sell", symbol="AAPL"):
            self.id, self.status = id, status
            self.filled_qty, self.filled_avg_price = filled_qty, filled_avg_price
            self.side, self.symbol = side, symbol

    class _Client:
        def __init__(self, statuses):
            self.statuses = list(statuses)
            self.submitted, self.cancels = [], []

        def submit_order(self, order_data):
            self.submitted.append(order_data)
            return TestAlpacaProtectiveStops._Order("stop-1", "new")

        def get_order_by_id(self, order_id):
            st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return st

        def cancel_order_by_id(self, order_id):
            self.cancels.append(order_id)

    def _broker(self, statuses):
        from src.execution.alpaca_broker import AlpacaBroker
        b = AlpacaBroker(api_key="x", secret_key="y", paper=True,
                         fill_poll_timeout_s=0.2, fill_poll_interval_s=0.01)
        b.client = self._Client(statuses)
        return b

    def test_places_a_day_stop_with_rounded_price(self):
        from alpaca.trading.enums import OrderSide, TimeInForce
        b = self._broker([None])
        oid = b.place_protective_stop("AAPL", Action.SELL, 2.123456, 98.7654)
        req = b.client.submitted[0]
        assert oid == "stop-1"
        assert req.side == OrderSide.SELL and req.time_in_force == TimeInForce.DAY
        assert float(req.stop_price) == pytest.approx(98.77) and float(req.qty) == pytest.approx(2.1235)

    def test_a_rejected_stop_returns_none_without_raising(self):
        b = self._broker([None])
        def boom(order_data):
            raise RuntimeError("stop price must be below market")
        b.client.submit_order = boom
        assert b.place_protective_stop("AAPL", Action.SELL, 1.0, 98.0) is None

    def test_cancel_reports_an_already_filled_stop(self):
        from alpaca.trading.enums import OrderStatus
        b = self._broker([self._Order("stop-1", OrderStatus.FILLED, 2.0, 98.5)])
        status, fill = b.cancel_protective_stop("stop-1", "t")
        assert status == "filled" and fill.price == pytest.approx(98.5) and fill.action == Action.SELL
        assert b.client.cancels == []

    def test_cancel_an_open_stop_waits_for_confirmation(self):
        from alpaca.trading.enums import OrderStatus
        b = self._broker([self._Order("stop-1", OrderStatus.NEW),
                          self._Order("stop-1", OrderStatus.CANCELED)])
        assert b.cancel_protective_stop("stop-1", "t") == ("canceled", None)
        assert b.client.cancels == ["stop-1"]

    def test_partially_filled_then_canceled_is_unknown(self):
        from alpaca.trading.enums import OrderStatus
        b = self._broker([self._Order("stop-1", OrderStatus.NEW),
                          self._Order("stop-1", OrderStatus.CANCELED, filled_qty=1.0)])
        assert b.cancel_protective_stop("stop-1", "t")[0] == "unknown"

    def test_cancel_that_never_confirms_is_unknown(self):
        from alpaca.trading.enums import OrderStatus
        b = self._broker([self._Order("stop-1", OrderStatus.NEW)])
        assert b.cancel_protective_stop("stop-1", "t")[0] == "unknown"

    def test_get_position_reads_no_position_as_flat(self):
        b = self._broker([None])
        class NotFound(Exception):
            status_code = 404
        def raise_(t):
            raise NotFound('{"code":40410000,"message":"position does not exist"}')
        b.client.get_open_position = raise_
        pos = b.get_position("AAPL")
        assert pos.quantity == 0.0 and not pos.unknown

    def test_get_position_reads_an_api_failure_as_unknown(self):
        b = self._broker([None])
        def raise_(t):
            raise ConnectionError("read timed out")
        b.client.get_open_position = raise_
        assert b.get_position("AAPL").unknown
