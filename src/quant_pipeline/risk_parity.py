"""Risk parity and equal-risk-contribution utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def risk_parity_weights(cov: pd.DataFrame) -> pd.Series:
    """Approximate risk parity weights from a covariance matrix."""

    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    w = inv_vol / inv_vol.sum()
    return pd.Series(w, index=cov.columns)


def rolling_risk_parity(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling risk parity weights over ``window`` observations."""

    weights = []
    index = []
    for i in range(window, len(returns) + 1):
        cov = returns.iloc[i - window : i].cov()
        w = risk_parity_weights(cov)
        weights.append(w)
        index.append(returns.index[i - 1])
    return pd.DataFrame(weights, index=index)


__all__ = ["risk_parity_weights", "rolling_risk_parity"]
