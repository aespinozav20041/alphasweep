import time
import numpy as np
import pandas as pd

from quant_pipeline.quality import quality_check
from quant_pipeline.decision import DecisionLoop
from quant_pipeline.simple_lstm import SimpleLSTM
from quant_pipeline.oms import OMS
from quant_pipeline.risk import RiskManager
from quant_pipeline.observability import Observability


class DummyExchange:
    def __init__(self):
        self.orders = []

    def create_order(self, symbol, side, qty, price, client_id, leverage=None):
        self.orders.append({"symbol": symbol, "side": side, "qty": qty})
        return "1"

    def cancel_order(self, order_id):
        pass

    def get_open_orders(self):
        return []


def test_quality_check_fields_and_freshness():
    now = time.time()
    bar = {
        "timestamp": now,
        "symbol": "BTC-USDT",
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 1.0,
    }
    assert quality_check(bar)
    bad = bar.copy()
    del bad["close"]
    assert not quality_check(bad)
    stale = bar.copy()
    stale["timestamp"] = now - 120
    assert not quality_check(stale)


def test_decision_loop_counts_quality_errors(tmp_path):
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
    train = pd.DataFrame({"ret": np.linspace(-0.01, 0.01, 20)})
    model.fit(train)
    loop = DecisionLoop(model, risk, oms, obs, ema_span=2, threshold=0.0, cooldown=0)
    bad_bar = {
        "timestamp": time.time(),
        "symbol": "BTC-USDT",
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        # missing close field to trigger quality error
        "volume": 1.0,
    }
    loop.on_bar(bad_bar)
    assert obs.quality_errors_total._value.get() == 1.0
    assert ex.orders == []
