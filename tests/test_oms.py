from quant_pipeline.oms import OMS, RateLimitError
import random
import pytest


class MockExchange:
    def __init__(self):
        self.orders = {}
        self.next_id = 1
        self.create_calls = 0

    def create_order(self, symbol, side, qty, price, client_id, leverage=None):
        self.create_calls += 1
        oid = str(self.next_id)
        self.next_id += 1
        self.orders[oid] = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "client_id": client_id,
            "filled": 0.0,
            "status": "NEW",
        }
        return oid

    def cancel_order(self, order_id):
        self.orders[order_id]["status"] = "CANCELED"

    def get_open_orders(self):
        res = []
        for oid, o in self.orders.items():
            if o["status"] in {"NEW", "PARTIAL"}:
                res.append(
                    {
                        "symbol": o["symbol"],
                        "side": o["side"],
                        "qty": o["qty"],
                        "price": o["price"],
                        "client_id": o["client_id"],
                        "order_id": oid,
                        "filled_qty": o["filled"],
                    }
                )
        return res

    def fill(self, order_id, qty):
        o = self.orders[order_id]
        o["filled"] += qty
        if o["filled"] >= o["qty"]:
            o["status"] = "FILLED"
        else:
            o["status"] = "PARTIAL"


class FlakyExchange(MockExchange):
    def __init__(self):
        super().__init__()
        self.fail_first = True
        self.attempts = 0

    def create_order(self, *a, **k):
        self.attempts += 1
        if self.fail_first:
            self.fail_first = False
            raise RateLimitError("rl")
        return super().create_order(*a, **k)


def _oms(exchange):
    info = {"BTC": {"tick_size": 1, "lot_size": 1, "min_notional": 10}}
    return OMS(exchange, info)


def test_state_transitions_and_idempotence():
    ex = FlakyExchange()
    oms = _oms(ex)
    order = oms.submit_order(symbol="BTC", side="buy", qty=1, price=100, client_id="a")
    assert ex.attempts == 2 and ex.create_calls == 1
    same = oms.submit_order(symbol="BTC", side="buy", qty=1, price=100, client_id="a")
    assert same is order
    assert ex.create_calls == 1
    ex.fill(order.order_id, 0.5)
    oms.handle_fill(order.order_id, 0.5)
    assert order.state == "partial"
    ex.fill(order.order_id, 0.5)
    oms.handle_fill(order.order_id, 0.5)
    assert order.state == "filled"


def test_cancel_replace():
    ex = MockExchange()
    oms = _oms(ex)
    order = oms.submit_order(symbol="BTC", side="buy", qty=2, price=100, client_id="b")
    old_id = order.order_id
    oms.replace_order("b", price=110, qty=1)
    assert ex.orders[old_id]["status"] == "CANCELED"
    assert order.order_id != old_id
    assert order.price == 110 and order.qty == 1
    assert order.state == "working"


def test_reconcile_no_duplicates():
    ex = MockExchange()
    oid = ex.create_order("BTC", "buy", 1, 100, "c")
    oms = _oms(ex)
    oms.reconcile()
    assert "c" in oms.orders
    assert oms.orders["c"].order_id == oid
    oms.reconcile()  # second reconcile should not duplicate
    assert len(oms.orders) == 1


class SimExchange(MockExchange):
    """Exchange simulator producing deterministic slippage."""

    def __init__(self, slippage_bps: float, seed: int = 0):
        super().__init__()
        self.slippage_bps = slippage_bps
        random.seed(seed)

    def execute(self, order_id: str):
        o = self.orders[order_id]
        slip = o["price"] * self.slippage_bps / 10_000
        fill_price = o["price"] + (slip if o["side"] == "buy" else -slip)
        o["filled"] = o["qty"]
        o["status"] = "FILLED"
        o["fill_price"] = fill_price
        return fill_price


def test_simulator_min_notional_cancel_replace_slippage():
    info = {"BTC": {"tick_size": 1, "lot_size": 0.1, "min_notional": 10}}
    ex = SimExchange(slippage_bps=50, seed=42)
    oms = OMS(ex, info)

    # minNotional enforcement
    with pytest.raises(ValueError):
        oms.submit_order(symbol="BTC", side="buy", qty=0.05, price=100, client_id="min")

    order = oms.submit_order(symbol="BTC", side="buy", qty=0.2, price=100, client_id="d")
    old = order.order_id
    oms.replace_order("d", price=105)
    assert ex.orders[old]["status"] == "CANCELED"

    fill_price = ex.execute(order.order_id)
    oms.handle_fill(order.order_id, order.qty)
    slip_bps = (fill_price - order.price) / order.price * 10_000
    assert abs(slip_bps - ex.slippage_bps) < 1e-6
    assert order.state == "filled"
