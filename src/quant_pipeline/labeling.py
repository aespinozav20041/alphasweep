"""Labeling utilities for supervised learning."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Compute forward returns over a given horizon.

    Parameters
    ----------
    df : DataFrame
        Must contain a ``close`` column representing the price.
    horizon : int
        Number of steps to look ahead.

    Returns
    -------
    Series
        The percentage return between ``close`` and ``close`` shifted by
        ``horizon`` steps forward. The final ``horizon`` rows will contain
        ``NaN`` as the future price is unknown.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    future = df["close"].shift(-horizon)
    current = df["close"]
    return ((future / current) - 1).rename("label")


def triple_barrier_labels(
    df: pd.DataFrame, upper: float, lower: float, horizon: int
) -> pd.Series:
    """Generate labels using the triple barrier method.

    The function inspects future ``high``/``low`` prices up to ``horizon`` steps
    and determines which barrier is touched first. Barriers are defined as
    multiples of the current ``close`` price.

    Parameters
    ----------
    df : DataFrame
        Must contain ``close``, ``high`` and ``low`` columns.
    upper, lower : float
        Multipliers applied to the ``close`` price to form the upper and lower
        barriers respectively.
    horizon : int
        Number of rows to look ahead for the vertical barrier.

    Returns
    -------
    Series
        Values are ``1`` when the upper barrier is hit first, ``-1`` when the
        lower barrier is hit and ``0`` if neither barrier is reached within the
        horizon.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    prices = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    labels = np.zeros(len(df), dtype=int)
    n = len(df)
    for i in range(n):
        up_price = prices[i] * (1 + upper)
        down_price = prices[i] * (1 - lower)
        end = min(i + horizon, n - 1)
        for j in range(i + 1, end + 1):
            if highs[j] >= up_price:
                labels[i] = 1
                break
            if lows[j] <= down_price:
                labels[i] = -1
                break
    return pd.Series(labels, index=df.index, name="label")


__all__ = ["forward_return", "triple_barrier_labels"]
