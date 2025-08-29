import pandas as pd

from ingest import IngestService
from quant_pipeline.observability import Observability
from quant_pipeline import ingest as qingest


def fake_orderbook(symbol, timeframe="1s", start=None, end=None):
    base = 1_600_000_000_000
    data = [
        {
            "timestamp": base,
            "bid1": 1.0,
            "ask1": 1.1,
            "bid_sz1": 2.0,
            "ask_sz1": 2.5,
            "trades_buy_vol": 1.0,
            "trades_sell_vol": 1.2,
        },
        {
            "timestamp": base + 1000,
            "bid1": 1.05,
            "ask1": 1.15,
            "bid_sz1": 2.1,
            "ask_sz1": 2.4,
            "trades_buy_vol": 0.5,
            "trades_sell_vol": 0.7,
        },
    ]
    return pd.DataFrame(data)


def fake_news(provider, symbols, start=None, end=None):
    base = 1_600_000_000_000
    rows = []
    for sym in symbols:
        rows.append({"timestamp": base, "sentiment": 0.1, "symbol": sym})
        rows.append({"timestamp": base + 60_000, "sentiment": -0.2, "symbol": sym})
    return pd.DataFrame(rows)


def test_ingest_orderbook(tmp_path, monkeypatch):
    obs = Observability()
    svc = IngestService(lake_path=tmp_path / "lake", observability=obs)
    svc.ingest_orderbook(fake_orderbook, ["BTC/USDT"], timeframe="1s")
    miss = obs.data_missing_bars_ratio.labels(symbol="BTC-USDT", timeframe="1s")._value.get()
    dup = obs.data_dup_ratio.labels(symbol="BTC-USDT")._value.get()
    lat = obs.latency_ms._sum.get()
    assert miss == 0.0
    assert dup == 0.0
    assert lat > 0.0


def test_ingest_news(tmp_path, monkeypatch):
    monkeypatch.setattr(qingest, "fetch_news_sentiment", fake_news)
    obs = Observability()
    svc = IngestService(lake_path=tmp_path / "lake", observability=obs)
    svc.ingest_news("http://api/{symbol}", ["BTC"], timeframe="1h")
    dup = obs.data_dup_ratio.labels(symbol="BTC")._value.get()
    lat = obs.latency_ms._sum.get()
    assert dup == 0.0
    assert lat > 0.0
