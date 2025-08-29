"""Feature engineering utilities."""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple features for training and backtesting.

    Currently this only computes simple returns but can be extended with more
    complex indicators. The input ``df`` must already be validated by
    :func:`quant_pipeline.doctor.validate_ohlcv`.
    """

    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out["ret"] = out["close"].pct_change().fillna(0.0)
    return out


__all__ = ["build_features"]
