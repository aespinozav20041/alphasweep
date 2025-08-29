"""Utility helpers including timer and exponential retry."""

import functools
import time
from contextlib import contextmanager
from typing import Callable, Tuple, Type, TypeVar

import pandas as pd

from .logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


@contextmanager
def timer(name: str):
    """Context manager measuring execution time."""

    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info(f"{name} took {duration:.3f}s")


def retry(
    exceptions: Tuple[Type[BaseException], ...],
    tries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry decorator with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    logger.warning(
                        "retry", extra={"exc": str(exc), "sleep": _delay, "tries": _tries}
                    )
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            return func(*args, **kwargs)

        return wrapper

    return decorator


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data and fix common data issues.

    The function aligns bars to the desired ``timeframe`` boundary, detects
    potential stock splits by looking at large gaps between consecutive bars,
    clips extreme outliers and returns a cleaned DataFrame with millisecond
    timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data containing at least ``timestamp``, ``open``, ``high``,
        ``low`` and ``close`` columns where ``timestamp`` is expressed in
        milliseconds.
    timeframe : str
        Pandas resampling rule such as ``"1h"`` or ``"1D"``.

    Returns
    -------
    pd.DataFrame
        Resampled and adjusted DataFrame with millisecond ``timestamp``.
    """

    if df.empty:
        return df

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out = out.set_index("timestamp").sort_index()

    # ------------------------------------------------------------------
    # detect and adjust stock splits using close/next open ratios
    # ------------------------------------------------------------------
    if {"open", "close"}.issubset(out.columns) and len(out) >= 2:
        ratio = out["close"].shift(1) / out["open"]
        split_points = ratio[(ratio > 1.5) | (ratio < 0.67)]
        if not split_points.empty:
            price_cols = [c for c in ["open", "high", "low", "close"] if c in out.columns]
            for ts, r in split_points.items():
                if r and r != 0:
                    out.loc[out.index < ts, price_cols] = out.loc[out.index < ts, price_cols].div(
                        r
                    )

    # ------------------------------------------------------------------
    # winsorise outliers to reduce the impact of bad ticks
    # ------------------------------------------------------------------
    if len(out) >= 10:
        for col in [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]:
            q_low = out[col].quantile(0.01)
            q_high = out[col].quantile(0.99)
            out[col] = out[col].clip(q_low, q_high)

    # ------------------------------------------------------------------
    # resample to regular timeframe boundaries
    # ------------------------------------------------------------------
    o = out["open"].resample(timeframe).first()
    h = out["high"].resample(timeframe).max()
    l = out["low"].resample(timeframe).min()
    c = out["close"].resample(timeframe).last()
    v = out.get("volume", pd.Series(dtype=float)).resample(timeframe).sum()
    out = pd.concat([o, h, l, c, v], axis=1, keys=["open", "high", "low", "close", "volume"])
    out = out.dropna(how="all").reset_index()
    out["timestamp"] = out["timestamp"].astype("int64") // 10 ** 6
    return out

