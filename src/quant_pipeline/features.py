"""Feature engineering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple features for training and backtesting.

    Computes a set of basic indicators such as returns, volatility, spreads and
    a simple microstructure proxy. The input ``df`` must already be validated by
    :func:`quant_pipeline.doctor.validate_ohlcv`.
    """

    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out["ret"] = out["close"].pct_change().fillna(0.0)
    # realised volatility over 5-bar window
    out["volatility"] = out["ret"].rolling(window=5).std()
    # high/low spread
    out["spread"] = (out["high"] - out["low"]) / out["close"]
    # price impact proxy
    out["price_impact"] = out["ret"].abs() / out["volume"].replace(0.0, np.nan)
    return out


@dataclass
class FeatureNormalizer:
    """Normalise features using statistics fitted on training data only."""

    feature_cols: Sequence[str]
    ffill_limit: int | None = None
    mean_: pd.Series | None = None
    std_: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> None:
        self.mean_ = df[self.feature_cols].mean()
        self.std_ = df[self.feature_cols].std().replace(0, 1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("normalizer must be fitted before calling transform")
        out = df.copy()
        if self.ffill_limit is not None:
            out[self.feature_cols] = out[self.feature_cols].ffill(
                limit=self.ffill_limit
            )
        out[self.feature_cols] = (out[self.feature_cols] - self.mean_) / self.std_
        return out


def generate_lstm_windows(
    df: pd.DataFrame, seq_len: int, feature_cols: Sequence[str]
) -> np.ndarray:
    """Return sliding windows shaped as ``[batch, seq_len, n_features]``."""

    data = df[feature_cols].to_numpy()
    if len(data) < seq_len:
        return np.empty((0, seq_len, len(feature_cols)))
    windows = [data[i : i + seq_len] for i in range(len(data) - seq_len + 1)]
    return np.stack(windows)


__all__ = ["build_features", "FeatureNormalizer", "generate_lstm_windows"]
