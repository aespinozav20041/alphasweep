"""Data quality utilities for validating and repairing OHLCV datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Type

import json
import pandas as pd

from .models import BarOHLCV
from .storage import LAKE_PATH, read_table, to_parquet

RUNS_PATH = Path(__file__).resolve().parents[2] / "runs"


def trading_calendar(start: int, end: int, timeframe: str) -> pd.DatetimeIndex:
    """Generate expected UTC bar timestamps between start and end inclusive."""
    start_ts = pd.to_datetime(start, unit="ms", utc=True)
    end_ts = pd.to_datetime(end, unit="ms", utc=True)
    return pd.date_range(start=start_ts, end=end_ts, freq=timeframe, inclusive="both")


def find_missing_bars(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Return DataFrame with missing bar ranges and their lengths."""
    existing = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    missing = calendar.difference(existing)
    if missing.empty:
        return pd.DataFrame(columns=["start", "end", "gap_len"])
    diff = missing.to_series().diff() != calendar.freq
    groups = diff.cumsum()
    gaps = (
        missing.to_series()
        .groupby(groups)
        .agg(["first", "last", "size"])
        .rename(columns={"first": "start", "last": "end", "size": "gap_len"})
    )
    return gaps.reset_index(drop=True)


def repair_missing_bars(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    method: str = "carry",
) -> pd.DataFrame:
    """Fill missing bars according to the chosen method."""
    df = df.set_index(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    df = df.reindex(calendar)

    if method == "drop":
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    elif method == "nan":
        pass
    else:  # carry
        df[["symbol", "timeframe", "source"]] = df[["symbol", "timeframe", "source"]].ffill()
        prev_close = df["close"].ffill()
        df["open"] = df["open"].fillna(prev_close)
        df["high"] = df["high"].fillna(prev_close)
        df["low"] = df["low"].fillna(prev_close)
        df["close"] = prev_close
        df["volume"] = df["volume"].fillna(0.0)

    df["timestamp"] = (df.index.view("int64") // 1_000_000).astype("int64")
    return df.reset_index(drop=True)


def clip_outliers(
    df: pd.DataFrame,
    price_z: float = 8.0,
    vol_z: float = 8.0,
) -> Tuple[pd.DataFrame, int]:
    """Winsorize price and volume columns with robust z-score."""
    df = df.copy()
    out_count = 0

    def _winsorize(series: pd.Series, z: float) -> Tuple[pd.Series, int]:
        med = series.median()
        mad = (series - med).abs().median()
        if mad == 0:
            return series, 0
        scale = 1.4826 * mad
        zscores = (series - med) / scale
        clip_lo = med - z * scale
        clip_hi = med + z * scale
        clipped = series.clip(clip_lo, clip_hi)
        cnt = ((zscores < -z) | (zscores > z)).sum()
        return clipped, int(cnt)

    for col in ["open", "high", "low", "close"]:
        df[col], cnt = _winsorize(df[col], price_z)
        out_count += cnt
    df["volume"], cnt = _winsorize(df["volume"], vol_z)
    out_count += cnt
    return df, out_count


@dataclass
class QualityReport:
    rows: int
    dup_ratio: float
    missing_bars: int
    outlier_count: int
    monotonic_ok: bool
    unique_index_ok: bool
    ts_range: Tuple[int, int]


def validate_contract(
    df: pd.DataFrame,
    schema_model: Type,
    *,
    missing_bars: int = 0,
    outlier_count: int = 0,
) -> QualityReport:
    """Validate dataset against schema and return quality metrics."""
    cols = set(schema_model.__fields__.keys())
    if cols - set(df.columns):
        raise ValueError("missing columns")
    key = ["timestamp", "symbol", "timeframe", "source"]
    dup = df.duplicated(subset=key)
    dup_ratio = dup.mean() if len(df) else 0.0
    monotonic_ok = df["timestamp"].is_monotonic_increasing
    unique_index_ok = not dup.any()
    ts_range: Tuple[int, int] = (
        int(df["timestamp"].min()) if len(df) else 0,
        int(df["timestamp"].max()) if len(df) else 0,
    )
    return QualityReport(
        rows=len(df),
        dup_ratio=float(dup_ratio),
        missing_bars=int(missing_bars),
        outlier_count=int(outlier_count),
        monotonic_ok=bool(monotonic_ok),
        unique_index_ok=bool(unique_index_ok),
        ts_range=ts_range,
    )


def doctor_ohlcv(
    symbol: str,
    timeframe: str,
    start: int,
    end: int,
    gap_threshold: float = 0.01,
    lake_path: Path = LAKE_PATH,
    runs_path: Path = RUNS_PATH,
) -> QualityReport:
    """Run data quality checks and repairs for OHLCV bars."""
    calendar = trading_calendar(start, end, timeframe)
    df = read_table("ohlcv", [symbol], start, end, timeframe, base_path=lake_path)
    df = df.sort_values("timestamp")

    missing = find_missing_bars(df, calendar)
    missing_total = int(missing["gap_len"].sum())
    gap_ratio = missing_total / len(calendar) if len(calendar) else 0.0
    if gap_ratio > gap_threshold:
        raise ValueError(f"gap ratio {gap_ratio:.2%} exceeds threshold {gap_threshold:.2%}")

    repaired = repair_missing_bars(df, calendar, method="carry")
    clipped, outliers = clip_outliers(repaired)
    report = validate_contract(
        clipped, BarOHLCV, missing_bars=missing_total, outlier_count=outliers
    )

    to_parquet(clipped, "ohlcv_clean", base_path=lake_path)

    rep_dir = runs_path / "reports" / "quality"
    rep_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"{symbol.replace('/', '_')}_{timeframe}_{start}_{end}.json"
    )
    with open(rep_dir / fname, "w") as f:
        json.dump(report.__dict__, f, indent=2)
    return report
