"""Label generation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier(df: pd.DataFrame, horizon: int, up_mult: float, down_mult: float) -> pd.Series:
    """Generate meta-labels using the triple barrier method.

    The function inspects future ``high``/``low`` prices up to ``horizon``
    steps and determines which barrier is touched first.

    Parameters
    ----------
    df : DataFrame
        Must contain ``close``, ``high`` and ``low`` columns. A ``volatility``
        column is optional and scales the barriers when present.
    horizon : int
        Number of rows to look ahead for the vertical barrier.
    up_mult, down_mult : float
        Multipliers applied to the volatility (or 1.0 when missing) to form the
        upper and lower barriers respectively.

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
    vol = df["volatility"].to_numpy() if "volatility" in df.columns else np.ones(len(df))
    labels = np.zeros(len(df), dtype=int)

    n = len(df)
    for i in range(n):
        upper = prices[i] * (1 + up_mult * vol[i])
        lower = prices[i] * (1 - down_mult * vol[i])
        end = min(i + horizon, n - 1)
        for j in range(i + 1, end + 1):
            if highs[j] >= upper:
                labels[i] = 1
                break
            if lows[j] <= lower:
                labels[i] = -1
                break
    return pd.Series(labels, index=df.index, name="label")


__all__ = ["triple_barrier"]
