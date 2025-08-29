import types
import pytest

from trading.brokers import alpaca, ibkr
from trading import ledger
from quant_pipeline.observability import Observability


class _AlpacaClient:
    def __init__(self, *, raise_error: bool = False, status: str = "accepted") -> None:
        self.raise_error = raise_error
        self.status = status

    def submit_order(self, **kwargs):  # pragma: no cover - behavior mocked in tests
        if self.raise_error:
            raise Exception("rate limit")
        return types.SimpleNamespace(id="a1", status=self.status)


class _IBClient:
    def __init__(self, *, raise_error: bool = False, status: str = "Submitted") -> None:
        self.raise_error = raise_error
        self.status = status

    def placeOrder(self, contract, order):  # pragma: no cover - behavior mocked in tests
        if self.raise_error:
            raise Exception("rate limit")
        trade = types.SimpleNamespace(
            order=types.SimpleNamespace(orderId="i1"),
            orderStatus=types.SimpleNamespace(status=self.status),
        )
        return trade


@pytest.mark.parametrize("broker,client_attr", [
    (alpaca, "_client"),
    (ibkr, "_client"),
])
def test_rate_limit(monkeypatch, broker, client_attr):
    ledger.clear()
    obs = Observability()
    if broker is alpaca:
        monkeypatch.setattr(broker, client_attr, lambda: _AlpacaClient(raise_error=True))
    else:
        monkeypatch.setattr(broker, client_attr, lambda: _IBClient(raise_error=True))
    with pytest.raises(Exception):
        broker.place_market_on_open("SPY", 100, obs)
    assert ledger.entries()[-1].status == "error"
    assert obs.order_errors_total._value.get() == 1


@pytest.mark.parametrize("broker,client_attr,cancel_status", [
    (alpaca, "_client", "canceled"),
    (ibkr, "_client", "Cancelled"),
])
def test_cancelled(monkeypatch, broker, client_attr, cancel_status):
    ledger.clear()
    obs = Observability()
    if broker is alpaca:
        monkeypatch.setattr(broker, client_attr, lambda: _AlpacaClient(status=cancel_status))
    else:
        monkeypatch.setattr(broker, client_attr, lambda: _IBClient(status=cancel_status))
    oid, status = broker.place_market_on_open("SPY", 100, obs)
    assert status == cancel_status
    entries = ledger.entries()
    assert entries[-1].order_id == oid
    assert entries[-1].status == cancel_status
    assert obs.order_errors_total._value.get() == 1
