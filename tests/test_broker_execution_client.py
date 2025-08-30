import time

from execution import BrokerExecutionClient, Order
from quant_pipeline.observability import Observability


def test_heartbeat_updates_observability():
    obs = Observability()
    client = BrokerExecutionClient(observability=obs, heartbeat_interval=0.1)
    time.sleep(0.25)
    assert obs._heartbeat_ts > 0
    client.close()


def test_disconnect_cancels_orders_and_reconnect(monkeypatch):
    obs = Observability()
    client = BrokerExecutionClient(observability=obs, heartbeat_interval=0.05)
    order = Order("SPY", 1, price=1.0)
    client.send(order)
    assert order.id in client._pending_orders

    state = {"called": False}

    def ping_fail_once():
        if not state["called"]:
            state["called"] = True
            return False
        return True

    monkeypatch.setattr(client, "_ping", ping_fail_once)
    time.sleep(0.15)

    assert order.id not in client._pending_orders
    assert obs.order_errors_total._value.get() == 1
    assert client._connected
    client.close()
