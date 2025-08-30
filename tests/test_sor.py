import datetime as dt

from engine import TradingEngine
from execution import ExecutionClient, Order
from risk import RiskManager, RiskLimits
from trading import sor


class StubClient(ExecutionClient):
    def __init__(self, price: float, volume: float, name: str):
        self.price = price
        self.volume = volume
        self.name = name
        self.sent: list[Order] = []

    def quote(self, symbol: str):
        return self.price, self.volume

    def send(self, order: Order) -> str:
        self.sent.append(order)
        return f"{self.name}-{len(self.sent)}"

    def cancel(self, order_id: str) -> None:
        pass

    def positions(self):
        return {}

    def clock(self):  # pragma: no cover - simple pass-through
        return dt.datetime.now(dt.timezone.utc)


def test_route_order_selects_by_price_and_volume():
    order = Order(symbol="BTC", qty=10)
    a = StubClient(price=100.0, volume=5, name="A")  # insufficient volume
    b = StubClient(price=101.0, volume=20, name="B")
    venues = {"A": a, "B": b}
    oid = sor.route_order(order, venues)
    assert oid.startswith("B-")
    assert not a.sent
    assert b.sent and b.sent[0] == order

    sell_order = Order(symbol="BTC", qty=-5)
    c = StubClient(price=100.0, volume=10, name="C")
    d = StubClient(price=110.0, volume=10, name="D")
    venues = {"C": c, "D": d}
    oid = sor.route_order(sell_order, venues)
    assert oid.startswith("D-")
    assert d.sent and d.sent[0] == sell_order


def test_trading_engine_uses_sor(monkeypatch):
    model = type("M", (), {"predict": lambda self, _: 1.0})()
    risk = RiskManager(RiskLimits(max_position=1000))
    client = StubClient(price=100, volume=100, name="X")
    engine = TradingEngine(model, lambda x: x, risk, client)

    called = {}

    def fake_route(order, venues):
        called["order"] = order
        return venues.send(order)

    monkeypatch.setattr(sor, "route_order", fake_route)

    class Data:
        symbol = "BTC"
        price = 100.0

    oid = engine.on_bar(Data())
    assert "order" in called
    assert oid.startswith("X-")
