import numpy as np
import pandas as pd
import pytest
import torch

from quant_pipeline.decision import DecisionLoop
from quant_pipeline.simple_lstm import SimpleLSTM
from quant_pipeline.oms import OMS
from quant_pipeline.risk import RiskManager
from quant_pipeline.observability import Observability


class DummyExchange:
    def __init__(self):
        self.orders = []
        self.reconcile_called = False

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
        self.reconcile_called = True
        return []


def test_decision_loop_sends_orders_and_reports_metrics(tmp_path):
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
    train = pd.DataFrame(
        {
            "ret": np.linspace(-0.05, 0.05, 100),
            "volatility": 0.0,
            "spread": 0.0,
            "mid_price": 0.0,
            "ob_imbalance": 0.0,
            "trade_imbalance": 0.0,
        }
    )
    model = SimpleLSTM(input_size=train.shape[1])
    model.fit(train)
    state_file = tmp_path / "state.pt"
    model.save_state(state_file)
    loop = DecisionLoop(
        model,
        risk,
        oms,
        obs,
        ema_span=2,
        threshold=0.0,
        cooldown=0,
        lstm_path=str(state_file),
    )

    bars = [
        {"timestamp": 1, "symbol": "BTC-USDT", "close": 100.0},
        {"timestamp": 2, "symbol": "BTC-USDT", "close": 101.0},
    ]
    for bar in bars:
        loop.on_bar(bar)

    assert len(ex.orders) == 1
    assert obs.orders_sent_total._value.get() == 1.0

    order_id = "1"
    order = ex.orders[0]
    loop.on_fill(order_id, order["qty"], order["price"])
    expected_pos = order["qty"] if order["side"] == "buy" else -order["qty"]
    assert loop.position["BTC-USDT"] == pytest.approx(expected_pos)

    loop.reconcile()
    assert ex.reconcile_called
    assert state_file.exists()
    saved = torch.load(state_file)
    assert "hidden" in saved
