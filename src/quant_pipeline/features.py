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
class FeatureBuilder:
    """Stateful feature builder operating on streaming bars.

    The builder keeps track of the previous close to compute returns for new
    bars on the fly. Each call to :meth:`update` returns a dataframe with the
    freshly created features for the provided bar.
    """

    def __init__(self) -> None:
        self._last_close: float | None = None

    def update(self, bar: dict) -> pd.DataFrame:
        close = float(bar["close"])
        ts = int(bar["timestamp"])
        ret = 0.0 if self._last_close is None else close / self._last_close - 1.0
        self._last_close = close
        return pd.DataFrame([{ "timestamp": ts, "ret": ret }])


class Scaler:
    """Running z-score scaler using Welford's algorithm.

    ``update`` consumes new observations and updates internal statistics while
    ``transform`` scales inputs using the statistics observed so far. When not
    enough data has been observed the scaled value is zero which makes this
    safe to use from the start of the stream.
    """

    def __init__(self) -> None:
        self.n = 0
        self.mean: pd.Series | None = None
        self.M2: pd.Series | None = None

    def update(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            if self.mean is None:
                self.mean = row.copy()
                self.M2 = pd.Series(0.0, index=row.index)
                self.n = 1
                continue
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.M2 += delta * (row - self.mean)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean is None:
            return pd.DataFrame(0.0, index=df.index, columns=df.columns)
        std = self.std().replace({0.0: 1.0})
        return (df - self.mean) / std

    def std(self) -> pd.Series:
        if self.mean is None or self.n < 2:
            return pd.Series(1.0, index=self.mean.index if self.mean is not None else [])
        return (self.M2 / (self.n - 1)).pow(0.5)


__all__ = ["build_features", "FeatureBuilder", "Scaler"]

