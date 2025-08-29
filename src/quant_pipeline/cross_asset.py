"""Cross-asset relationship utilities."""

from __future__ import annotations

import pandas as pd


def pair_spread(df: pd.DataFrame, asset_a: str, asset_b: str) -> pd.Series:
    """Return price spread between two assets.

    Parameters
    ----------
    df : DataFrame
        Must contain columns for ``asset_a`` and ``asset_b`` prices.
    asset_a, asset_b : str
        Column names representing the two assets.
    """

    return df[asset_a] - df[asset_b]


class CrossAssetSignal:
    """Generate cross-asset signals for multiple pairs."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = {}
        for a, b in self.pairs:
            signals[f"{a}_{b}_spread"] = pair_spread(df, a, b)
        return pd.DataFrame(signals, index=df.index)


__all__ = ["pair_spread", "CrossAssetSignal"]
