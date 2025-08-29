from quant_pipeline.decision import DecisionLoop
from quant_pipeline.simple_lstm import SimpleLSTM
from quant_pipeline.oms import OMS
from quant_pipeline.risk import RiskManager
from quant_pipeline.observability import Observability


class DummyExchange:
    def __init__(self):
        self.orders = []

    def create_order(self, symbol, side, qty, price, client_id, leverage=None):
        self.orders.append({
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "client_id": client_id,
        })
        return str(len(self.orders))

    def cancel_order(self, order_id):
        pass

    def get_open_orders(self):
        return []


def test_decision_loop_sends_orders_and_reports_metrics():
    ex = DummyExchange()
    oms = OMS(ex, {"BTC-USDT": {"tick_size": 1, "lot_size": 0, "min_notional": 0}})
    risk = RiskManager(
        max_dd_daily=1.0,
        max_dd_weekly=1.0,
        latency_threshold=1000,
        latency_window=1,
        pause_minutes=1,
    )
    obs = Observability()
    model = SimpleLSTM()
    loop = DecisionLoop(model, risk, oms, obs, ema_span=2, threshold=0.0, cooldown=0)

    bars = [
        {"timestamp": 1, "symbol": "BTC-USDT", "close": 100.0},
        {"timestamp": 2, "symbol": "BTC-USDT", "close": 101.0},
    ]
    for bar in bars:
        loop.on_bar(bar)

    assert len(ex.orders) == 1
    assert obs.orders_sent_total._value.get() == 1.0
