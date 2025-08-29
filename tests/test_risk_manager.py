from risk import RiskLimits, RiskManager
from execution import Order


def test_trailing_stop_kill_switch():
    limits = RiskLimits(max_position=1, trailing_pct=0.1, max_drawdown=0.2)
    rm = RiskManager(limits)
    order = Order(symbol="XYZ", qty=1, price=100.0)
    rm.post_trade(order, {"XYZ": 1})
    rm.update_price("XYZ", 105.0)
    assert not rm._kill
    rm.update_price("XYZ", 93.0)
    assert rm._kill
