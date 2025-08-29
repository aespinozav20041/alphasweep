"""Data quality checks for OHLCV datasets."""

from __future__ import annotations

import pandas as pd


def validate_ohlcv(df: pd.DataFrame, timeframe_ms: int, *, allow_gap_ratio: float = 0.0) -> pd.DataFrame:
    """Validate OHLCV data integrity.

    Parameters
    ----------
    df: DataFrame
        OHLCV records sorted by timestamp.
    timeframe_ms: int
        Expected bar size in milliseconds.
    allow_gap_ratio: float, optional
        Maximum fraction of missing bars allowed before raising an error.

    Returns
    -------
    DataFrame
        The validated (and timestamp-sorted) DataFrame.
    """

    if df.empty:
        raise ValueError("no data to validate")

    df = df.sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps detected")

    diffs = df["timestamp"].diff().dropna()
    if (diffs <= 0).any():
        raise ValueError("timestamps not strictly increasing")

    expected = timeframe_ms
    missing = (diffs > expected).sum()
    ratio = missing / max(len(diffs), 1)
    if ratio > allow_gap_ratio:
        raise ValueError(f"missing bars ratio {ratio:.1%} exceeds {allow_gap_ratio:.1%}")
    return df


__all__ = ["validate_ohlcv"]
