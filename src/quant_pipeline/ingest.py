"""Market data ingestion routines."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import ccxt
import pandas as pd

from .logging import get_logger
from .storage import to_parquet
from .utils import retry

logger = get_logger(__name__)

# default locations
DATA_PATH = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DB = DATA_PATH / "ingest.db"


def _tf_to_ms(timeframe: str) -> int:
    units = {"m": 60, "h": 60 * 60, "d": 24 * 60 * 60}
    unit = timeframe[-1]
    if unit not in units:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return int(timeframe[:-1]) * units[unit] * 1000


def _get_watermark(conn: sqlite3.Connection, exchange: str, symbol: str, timeframe: str) -> Optional[int]:
    cur = conn.execute(
        "SELECT last_ts FROM ingest_meta WHERE exchange=? AND symbol=? AND timeframe=?",
        (exchange, symbol, timeframe),
    )
    row = cur.fetchone()
    return int(row[0]) if row else None


def _set_watermark(conn: sqlite3.Connection, exchange: str, symbol: str, timeframe: str, ts: int) -> None:
    conn.execute(
        """
        INSERT INTO ingest_meta(exchange, symbol, timeframe, last_ts)
        VALUES(?,?,?,?)
        ON CONFLICT(exchange, symbol, timeframe)
        DO UPDATE SET last_ts=excluded.last_ts
        """,
        (exchange, symbol, timeframe, int(ts)),
    )
    conn.commit()


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_meta (
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            last_ts INTEGER NOT NULL,
            PRIMARY KEY(exchange, symbol, timeframe)
        )
        """
    )
    conn.commit()
    return conn


def ingest_ohlcv_ccxt(
    exchange: str,
    symbols: Iterable[str],
    timeframe: str = "1h",
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 1000,
    out: Path = Path(__file__).resolve().parents[2] / "lake",
    db_path: Path = DEFAULT_DB,
) -> None:
    """Ingest OHLCV data from a CCXT exchange and persist to the data lake."""

    ex_cls = getattr(ccxt, exchange)
    ex = ex_cls({"enableRateLimit": True})
    tf_ms = _tf_to_ms(timeframe)
    now = until or ex.milliseconds()

    conn = _init_db(db_path)

    for sym in symbols:
        canon_sym = sym.replace("/", "-")
        start = since or 0
        wm = _get_watermark(conn, exchange, canon_sym, timeframe)
        if wm is not None and start <= wm:
            start = wm + tf_ms
        all_batches = []
        cursor = start

        @retry((ccxt.NetworkError, ccxt.DDoSProtection, ccxt.RequestTimeout))
        def _fetch(s: int):
            return ex.fetch_ohlcv(sym, timeframe=timeframe, since=s, limit=limit)

        while cursor < now:
            batch = _fetch(cursor)
            if not batch:
                break
            df = pd.DataFrame(
                batch, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df = df.astype(
                {
                    "timestamp": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                }
            )
            df["volume"] = df["volume"].clip(lower=0.0)
            df["symbol"] = canon_sym
            df["source"] = exchange
            df["timeframe"] = timeframe
            all_batches.append(df)
            last_ts = int(df["timestamp"].max())
            cursor = last_ts + tf_ms

        if not all_batches:
            continue
        out_df = pd.concat(all_batches, ignore_index=True)
        out_df = out_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        if not out_df["timestamp"].is_monotonic_increasing:
            raise ValueError("timestamps not monotonic")
        to_parquet(out_df, "ohlcv", base_path=out)
        _set_watermark(conn, exchange, canon_sym, timeframe, int(out_df["timestamp"].max()))

    conn.close()

