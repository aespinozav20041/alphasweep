import pytest
import torch

from quant_pipeline.simple_lstm import SimpleLSTM
from quant_pipeline.decision import DecisionLoop
from quant_pipeline.risk import RiskManager
from quant_pipeline.oms import OMS
from quant_pipeline.observability import Observability


class DummyExchange:
    def create_order(self, symbol, side, qty, price, client_id, leverage=None):
        return "id"

    def cancel_order(self, order_id):
        pass

    def get_open_orders(self):
        return []


def make_loop(path):
    model = SimpleLSTM()
    risk = RiskManager(
        max_dd_daily=1.0,
        max_dd_weekly=1.0,
        latency_threshold=1e9,
        latency_window=1,
        pause_minutes=1,
    )
    oms = OMS(DummyExchange(), {})
    obs = Observability()
    return DecisionLoop(
        model,
        risk,
        oms,
        obs,
        threshold=10.0,
        snapshot_path=str(path),
    )


def test_state_persistence(tmp_path):
    snap = tmp_path / "snap.pkl"
    bar1 = {"timestamp": 1, "symbol": "BTC", "close": 100.0}
    bar2 = {"timestamp": 2, "symbol": "BTC", "close": 100.0}

    loop1 = make_loop(snap)
    loop1.on_bar(bar1)
    loop1.position["BTC"] = 1.23
    loop1.save()

    loop2 = make_loop(snap)
    assert loop2.position == {"BTC": 1.23}
    assert loop2.scaler.n == loop1.scaler.n
    assert loop2.model.hidden is not None
    for h1, h2 in zip(loop1.model.hidden, loop2.model.hidden):
        assert torch.allclose(h1, h2)

    loop1.on_bar(bar2)
    loop2.on_bar(bar2)
    assert loop1._ema == pytest.approx(loop2._ema)
