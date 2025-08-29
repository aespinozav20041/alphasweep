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


def _prep_ts(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Ensure timestamp column is datetime with tz and set as index."""

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    return df.set_index("timestamp").sort_index()


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to the specified rule."""

    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df.get("volume", pd.Series(dtype=float)).resample(rule).sum()
    return pd.concat([o, h, l, c, v], axis=1, keys=["open", "high", "low", "close", "volume"]).dropna(how="all")


def _adjust_splits_outliers(df: pd.DataFrame, corporate: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Adjust price history for splits and winsorise outliers."""

    out = df.copy()
    if corporate is not None and not corporate.empty and "split_ratio" in corporate.columns:
        corp = corporate.copy()
        corp["timestamp"] = pd.to_datetime(corp["timestamp"], unit="ms", utc=True)
        corp = corp.sort_values("timestamp")
        split = corp.set_index("timestamp")["split_ratio"]
        split = split.reindex(out.index, method="bfill").fillna(1.0)
        adj = split.iloc[::-1].cumprod().iloc[::-1]
        price_cols = [c for c in ["open", "high", "low", "close"] if c in out.columns]
        out[price_cols] = out[price_cols].div(adj, axis=0)
    # simple outlier handling using winsorisation
    if len(out) >= 10:
        for col in [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]:
            q_low = out[col].quantile(0.01)
            q_high = out[col].quantile(0.99)
            out[col] = out[col].clip(q_low, q_high)
    return out


def combine_market_data(
    ohlcv: pd.DataFrame,
    l2: Optional[pd.DataFrame] = None,
    l3: Optional[pd.DataFrame] = None,
    corporate: Optional[pd.DataFrame] = None,
    macro: Optional[pd.DataFrame] = None,
    *,
    tz: str = "UTC",
    resample_rule: str = "1h",
) -> pd.DataFrame:
    """Merge OHLCV with L2/L3, corporate actions and macro data.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Standard OHLCV data with millisecond timestamps.
    l2, l3, corporate, macro : pd.DataFrame, optional
        Additional datasets indexed by millisecond timestamps.
    tz : str
        Target timezone for the resulting index.
    resample_rule : str
        Pandas resampling rule, e.g. "1h" or "15min".
    """

    base = _prep_ts(ohlcv, tz)
    base = _adjust_splits_outliers(base, corporate)
    base = _resample_ohlcv(base, resample_rule)

    def _merge(df: Optional[pd.DataFrame], how: str = "mean") -> pd.DataFrame:
        if df is None or df.empty:
            return base
        other = _prep_ts(df, tz)
        if how == "mean":
            other = other.resample(resample_rule).mean()
        else:
            other = other.resample(resample_rule).last()
        return base.join(other, how="left")

    base = _merge(l2)
    base = _merge(l3)
    if macro is not None and not macro.empty:
        macro_p = _prep_ts(macro, tz).resample(resample_rule).last().ffill()
        base = base.join(macro_p, how="left")

    return base.reset_index()


__all__ = ["ingest_ohlcv_ccxt", "combine_market_data"]

