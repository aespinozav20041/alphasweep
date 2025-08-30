"""Macro factor calculations such as volatility spreads and carry."""

from __future__ import annotations

import pandas as pd


def vol_spread(implied: pd.Series, realized: pd.Series) -> pd.Series:
    """Difference between implied and realized volatility."""
    return implied - realized


def carry(spot: pd.Series, future: pd.Series, days: int) -> pd.Series:
    """Compute simple carry given spot and future prices."""
    return (future - spot) / days


def inflation_adjusted_rate(rate: pd.Series, inflation: pd.Series) -> pd.Series:
    """Return real rate after inflation."""
    return rate - inflation


__all__ = ["vol_spread", "carry", "inflation_adjusted_rate"]
