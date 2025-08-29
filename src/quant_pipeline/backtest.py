"""Very small backtesting helpers used for tests."""

from __future__ import annotations

import pandas as pd


def run_backtest(df: pd.DataFrame, *, threshold: float = 0.0) -> float:
    """Run a naive backtest.

    Parameters
    ----------
    df: DataFrame
        Must contain a ``ret`` column representing returns.
    threshold: float
        A simple trading rule: go long if the return is above ``threshold`` and
        short otherwise. This keeps the implementation intentionally
        lightweight for testing purposes.

    Returns
    -------
    float
        The cumulative return of the strategy.
    """

    if "ret" not in df.columns:
        raise ValueError("missing ret column")
    signal = (df["ret"] > threshold).astype(int) * 2 - 1
    pnl = (signal.shift().fillna(0) * df["ret"]).cumsum().iloc[-1]
    return float(pnl)


__all__ = ["run_backtest"]
