"""Helpers for writing and reading partitioned parquet tables and SQLite logs."""

from pathlib import Path
from typing import Iterable, Optional, Sequence, Type
import sqlite3

import pandas as pd

from .models import (
    BarOHLCV,
    OrderBookBest,
    PerpMetrics,
    NewsSentiment,
    CorporateAction,
    MacroIndicator,
)

# Base directory for lake storage
LAKE_PATH = Path(__file__).resolve().parents[2] / "lake"

# SQLite database for runtime trading logs
DATA_PATH = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DATA_PATH / "trading.db"
_sql_conn: sqlite3.Connection | None = None


def _db_conn(path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection creating required tables on first use."""

    global _sql_conn
    if _sql_conn is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _sql_conn = sqlite3.connect(path)
        _sql_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                symbol TEXT NOT NULL,
                ts INTEGER NOT NULL,
                yhat REAL NOT NULL,
                p_long REAL NOT NULL,
                p_short REAL NOT NULL
            )
            """
        )
        _sql_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                order_id TEXT PRIMARY KEY,
                ts INTEGER NOT NULL,
                price REAL NOT NULL,
                qty REAL NOT NULL,
                fees REAL NOT NULL,
                slippage REAL NOT NULL
            )
            """
        )
        _sql_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS equity (
                ts INTEGER PRIMARY KEY,
                nav REAL NOT NULL,
                cash REAL NOT NULL,
                exposure REAL NOT NULL
            )
            """
        )
        _sql_conn.commit()
    return _sql_conn

# Mapping between table names and their schemas.  Historically the OHLCV table
# was referenced as either ``ohlcv`` or ``bar_ohlcv`` so we keep both aliases to
# remain backward compatible.
SCHEMAS = {
    "ohlcv": BarOHLCV,
    "bar_ohlcv": BarOHLCV,
    "orderbook_best": OrderBookBest,
    "perp_metrics": PerpMetrics,
    "news": NewsSentiment,
    "corporate": CorporateAction,
    "macro": MacroIndicator,
}


def _check_columns_index(df: pd.DataFrame, model: Type) -> None:
    """Ensure required columns exist and index rules are met."""

    missing = set(model.__fields__.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    index = pd.MultiIndex.from_frame(df[["timestamp", "symbol", "timeframe", "source"]])
    if not index.is_unique:
        raise ValueError("index not unique")
    if not index.is_monotonic_increasing:
        raise ValueError("index not monotonically increasing")


def _validate_records(df: pd.DataFrame, model: Type) -> None:
    """Run Pydantic validation for each row."""

    for rec in df.to_dict("records"):
        model(**rec)


def to_parquet(
    table_df: pd.DataFrame,
    table_name: str,
    partitioning: Optional[Sequence[str]] = ("timeframe", "symbol", "dt"),
    base_path: Path = LAKE_PATH,
) -> None:
    """Write DataFrame to a partitioned parquet dataset."""

    model = SCHEMAS[table_name]
    df = table_df.copy()

    _check_columns_index(df, model)

    if model is BarOHLCV:
        if df[["open", "high", "low", "close"]].isnull().any().any():
            raise ValueError("NaN in OHLC fields")
        df["volume"] = df["volume"].fillna(0.0)

    _validate_records(df, model)

    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")

    dest = base_path / table_name
    dest.mkdir(parents=True, exist_ok=True)

    df.to_parquet(dest, engine="pyarrow", index=False, partition_cols=list(partitioning))


def read_table(
    table_name: str,
    symbols: Iterable[str],
    start: int,
    end: int,
    timeframe: str,
    base_path: Path = LAKE_PATH,
) -> pd.DataFrame:
    """Read a table from the lake with basic filtering."""

    path = base_path / table_name
    start_dt = pd.to_datetime(start, unit="ms", utc=True).strftime("%Y-%m-%d")
    end_dt = pd.to_datetime(end, unit="ms", utc=True).strftime("%Y-%m-%d")

    filters = [
        ("timeframe", "=", timeframe),
        ("symbol", "in", list(symbols)),
        ("dt", ">=", start_dt),
        ("dt", "<=", end_dt),
    ]

    df = pd.read_parquet(path, filters=filters, engine="pyarrow")
    model = SCHEMAS[table_name]
    _check_columns_index(df, model)
    _validate_records(df, model)
    df = df.drop(columns=["dt"]) if "dt" in df.columns else df
    cols = list(model.__fields__.keys())
    for col in cols:
        typ = model.__fields__[col].type_
        if typ is int:
            df[col] = df[col].astype("int64")
        elif typ is float:
            df[col] = df[col].astype("float64")
        else:
            df[col] = df[col].astype(str)
    return df[cols]


# ---------------------------------------------------------------------------
# SQLite logging helpers
# ---------------------------------------------------------------------------

def record_prediction(
    symbol: str,
    ts: int,
    yhat: float,
    p_long: float,
    p_short: float,
    db_path: Path = DB_PATH,
) -> None:
    """Persist a model prediction."""

    conn = _db_conn(db_path)
    conn.execute(
        "INSERT INTO predictions(symbol, ts, yhat, p_long, p_short) VALUES (?,?,?,?,?)",
        (symbol, int(ts), float(yhat), float(p_long), float(p_short)),
    )
    conn.commit()


def record_fill(
    order_id: str,
    ts: int,
    price: float,
    qty: float,
    fees: float,
    slippage: float,
    db_path: Path = DB_PATH,
) -> None:
    """Persist a fill record."""

    conn = _db_conn(db_path)
    conn.execute(
        "INSERT INTO fills(order_id, ts, price, qty, fees, slippage) VALUES (?,?,?,?,?,?)",
        (order_id, int(ts), float(price), float(qty), float(fees), float(slippage)),
    )
    conn.commit()


def record_equity(
    ts: int,
    nav: float,
    cash: float,
    exposure: float,
    db_path: Path = DB_PATH,
) -> None:
    """Persist an equity snapshot."""

    conn = _db_conn(db_path)
    conn.execute(
        "INSERT INTO equity(ts, nav, cash, exposure) VALUES (?,?,?,?)",
        (int(ts), float(nav), float(cash), float(exposure)),
    )
    conn.commit()
