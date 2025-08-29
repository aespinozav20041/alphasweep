"""Helpers for writing and reading partitioned parquet tables."""

from pathlib import Path
from typing import Iterable, Optional, Sequence, Type

import pandas as pd

from .models import BarOHLCV, OrderBookBest, PerpMetrics

# Base directory for lake storage
LAKE_PATH = Path(__file__).resolve().parents[2] / "lake"

# Mapping between table names and their schemas.  Historically the OHLCV table
# was referenced as either ``ohlcv`` or ``bar_ohlcv`` so we keep both aliases to
# remain backward compatible.
SCHEMAS = {
    "ohlcv": BarOHLCV,
    "bar_ohlcv": BarOHLCV,
    "orderbook_best": OrderBookBest,
    "perp_metrics": PerpMetrics,
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
