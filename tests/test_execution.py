from execution import BrokerExecutionClient, CostModel, Order


def test_broker_cancel_on_disconnect(capsys):
    client = BrokerExecutionClient(cost_model=CostModel())
    o1 = Order(symbol="XYZ", qty=1, price=100.0)
    o2 = Order(symbol="XYZ", qty=-1, price=101.0)
    client.send(o1)
    client.send(o2)
    assert len(client._orders) == 2
    client.handle_disconnect()
    assert len(client._orders) == 0
