import yaml
from pathlib import Path

from quant_pipeline.oms import OMS
from quant_pipeline.decision import SignalToOrdersMapper


class DummyExchange:
    def create_order(self, *a, **k):
        return "1"

    def cancel_order(self, order_id):
        pass

    def get_open_orders(self):
        return []


def _oms():
    info = {"BTC": {"tick_size": 1, "lot_size": 1, "min_notional": 0}}
    return OMS(DummyExchange(), info)


def test_schedule_child_orders_strategies():
    oms = _oms()
    twap = oms.schedule_child_orders(symbol="BTC", side="buy", qty=100, strategy="twap", intervals=4)
    assert twap == [25.0, 25.0, 25.0, 25.0]
    vols = [100, 200, 200, 500]
    vwap = oms.schedule_child_orders(symbol="BTC", side="buy", qty=100, strategy="vwap", volume_profile=vols)
    assert vwap == [10.0, 20.0, 20.0, 50.0]
    pov = oms.schedule_child_orders(
        symbol="BTC",
        side="buy",
        qty=100,
        strategy="pov",
        volume_profile=vols,
        participation=0.1,
    )
    assert pov == [10.0, 20.0, 20.0, 50.0]


def test_signal_to_orders_mapper_generates_orders():
    oms = _oms()
    mapper = SignalToOrdersMapper(oms, strategy="twap", intervals=2)
    orders = mapper.generate_orders("BTC", price=100, current_position=0, target_position=10)
    assert len(orders) == 2
    assert sum(o["qty"] for o in orders) == 10
    assert all(o["side"] == "buy" for o in orders)


def test_execution_config_smoke():
    cfg = yaml.safe_load(Path("conf/execution.yaml").read_text())
    assert "participation" in cfg and "vwap_window" in cfg
