from trading.brokers import mock
from trading import ledger


def test_mock_market_on_open_records_ledger():
    ledger.clear()
    oid, status = mock.place_market_on_open("SPY", 123.45)
    assert status == "filled"
    assert oid.startswith("mock-")
    assert any(e.order_id == oid and e.ticker == "SPY" for e in ledger.entries)


def test_min_notional_enforced():
    ledger.clear()
    try:
        mock.place_market_on_open("SPY", 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("minNotional not enforced")
