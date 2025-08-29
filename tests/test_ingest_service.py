import ccxt
import time

from ingest import IngestService, schedule_job
from quant_pipeline.observability import Observability


class FakeExchange:
    def __init__(self, data):
        self.data = data

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        return [row for row in self.data if row[0] >= since][: limit or len(self.data)]

    def milliseconds(self):
        return max(row[0] for row in self.data) + 3_600_000


def test_service_records_metrics(tmp_path, monkeypatch):
    base = 1_600_000_000_000
    tf = 3_600_000
    data = [
        [base, 1, 2, 0, 1.5, 1],
        [base + tf, 1.5, 2.5, 1, 2, 2],
        [base + 2 * tf, 2, 3, 1.5, 2.5, 3],
    ]

    lake = tmp_path / "lake"
    fake = FakeExchange(data)
    monkeypatch.setattr(ccxt, "binance", lambda cfg: fake)

    obs = Observability()
    svc = IngestService(lake_path=lake, observability=obs)
    svc.ingest_ccxt(
        "binance",
        ["BTC/USDT"],
        timeframe="1h",
        since=base,
        until=base + 2 * tf,
        db_path=lake / "ingest.db",
    )

    missing = obs.data_missing_bars_ratio.labels(symbol="BTC-USDT", timeframe="1h")._value.get()
    dup = obs.data_dup_ratio.labels(symbol="BTC-USDT")._value.get()
    latency = obs.latency_ms._sum.get()
    assert missing == 0.0
    assert dup == 0.0
    assert latency > 0.0


def test_schedule_job_runs(monkeypatch):
    calls = []

    def job():
        calls.append(time.time())

    stop = schedule_job(0.01, job)
    time.sleep(0.05)
    stop.set()
    assert len(calls) >= 2

