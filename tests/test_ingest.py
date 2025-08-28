import sqlite3
from pathlib import Path

import ccxt

from quant_pipeline.ingest import ingest_ohlcv_ccxt
from quant_pipeline.storage import read_table


class FakeExchange:
    def __init__(self, data, fail_first=False):
        self.data = data
        self.fail_first = fail_first
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise ccxt.NetworkError("temporary")
        result = [row for row in self.data if row[0] >= since][: limit or len(self.data)]
        return result

    def milliseconds(self):
        return max(row[0] for row in self.data) + 3_600_000


def test_ingest_retry_and_resume(tmp_path, monkeypatch):
    base = 1_600_000_000_000
    tf = 3_600_000
    data = [
        [base, 1, 2, 0, 1.5, 1],
        [base + tf, 1.5, 2.5, 1, 2, 2],
        [base + 2 * tf, 2, 3, 1.5, 2.5, 3],
    ]

    lake = tmp_path / "lake"
    db = tmp_path / "ingest.db"

    first = FakeExchange(data, fail_first=True)
    monkeypatch.setattr(ccxt, "binance", lambda cfg: first)
    ingest_ohlcv_ccxt(
        "binance",
        ["BTC/USDT"],
        timeframe="1h",
        since=base,
        until=base + tf,
        out=lake,
        db_path=db,
    )
    assert first.calls >= 2

    second = FakeExchange(data)
    monkeypatch.setattr(ccxt, "binance", lambda cfg: second)
    ingest_ohlcv_ccxt(
        "binance",
        ["BTC/USDT"],
        timeframe="1h",
        since=base,
        until=base + 2 * tf,
        out=lake,
        db_path=db,
    )

    df = read_table(
        "ohlcv",
        symbols=["BTC-USDT"],
        start=base,
        end=base + 2 * tf,
        timeframe="1h",
        base_path=lake,
    )
    assert len(df) == 3
    assert df["timestamp"].is_monotonic_increasing

    conn = sqlite3.connect(db)
    ts = conn.execute(
        "SELECT last_ts FROM ingest_meta WHERE exchange=? AND symbol=? AND timeframe=?",
        ("binance", "BTC-USDT", "1h"),
    ).fetchone()[0]
    assert ts == base + 2 * tf
