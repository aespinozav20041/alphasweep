"""Feature engineering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, vol_window: int = 10) -> pd.DataFrame:
    """Create engineered features for training and backtesting.

    Parameters
    ----------
    df:
        Input OHLCV dataframe validated by
        :func:`quant_pipeline.doctor.validate_ohlcv`.
    vol_window:
        Window size used for the rolling volatility indicator.
    """

    out = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Returns
    out["ret"] = out["close"].pct_change().fillna(0.0)

    # Volatility indicator using rolling standard deviation of returns.
    out["vol"] = out["ret"].rolling(vol_window, min_periods=1).std().fillna(0.0)

    # Price based spreads.
    out["hl_spread"] = (out["high"] - out["low"]) / out["close"]
    out["oc_spread"] = (out["close"] - out["open"]) / out["open"]

    # Simple microstructure proxy: position of the close within the bar range.
    rng = out["high"] - out["low"]
    out["pos_in_range"] = (out["close"] - out["low"]) / rng.replace({0.0: np.nan})
    out["pos_in_range"] = out["pos_in_range"].fillna(0.0)

    return out


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


def causal_normalize(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Causally z-score normalise a dataframe.

    The statistics used for normalisation only consider past values to avoid
    look-ahead bias. Missing values are forward filled up to ``limit`` steps to
    retain continuity without leaking too far into the future.
    """

    filled = df.ffill(limit=limit)
    mean = filled.expanding(min_periods=1).mean().shift().fillna(0.0)
    std = (
        filled.expanding(min_periods=1).std().shift().fillna(1.0).replace({0.0: 1.0})
    )
    normalised = (filled - mean) / std
    return normalised.fillna(0.0)


def sliding_window_tensor(
    df: pd.DataFrame, seq_len: int, *, stride: int = 1
) -> np.ndarray:
    """Generate sliding window tensors for LSTM training.

    Returns an array with shape ``[batch, seq_len, n_features]`` where ``batch``
    is the number of windows extracted from the dataframe using the provided
    ``seq_len`` and ``stride``.
    """

    data = df.to_numpy()
    if len(data) < seq_len:
        return np.empty((0, seq_len, data.shape[1]))

    windows = [data[i : i + seq_len] for i in range(0, len(data) - seq_len + 1, stride)]
    return np.stack(windows, axis=0)


__all__ = [
    "build_features",
    "FeatureBuilder",
    "Scaler",
    "causal_normalize",
    "sliding_window_tensor",
]
