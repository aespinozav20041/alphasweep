"""Feature engineering utilities."""

from __future__ import annotations

from typing import Iterable, Sequence

from collections import deque
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature set for training and backtesting.

    The input ``df`` must already be validated by
    :func:`quant_pipeline.doctor.validate_ohlcv` and may optionally contain
    orderbook/trade statistics.  Besides simple returns this function adds
    volatility, spread and microstructure indicators when the necessary columns
    are present.
    """

    out = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Basic return and volatility features
    out["ret"] = out["close"].pct_change().fillna(0.0)
    out["volatility"] = (
        out["ret"].rolling(window=20, min_periods=1).std().fillna(0.0)
    )

    # Spread indicators from order book or high/low fallback
    if {"bid1", "ask1"}.issubset(out.columns):
        out["spread"] = out["ask1"] - out["bid1"]
        out["mid_price"] = (out["ask1"] + out["bid1"]) / 2.0
    else:
        out["spread"] = out["high"] - out["low"]
        out["mid_price"] = out["close"]

    # Order book imbalance as microstructure feature
    if {"bid_sz1", "ask_sz1"}.issubset(out.columns):
        denom = (out["bid_sz1"] + out["ask_sz1"]).replace(0, np.nan)
        out["ob_imbalance"] = ((out["bid_sz1"] - out["ask_sz1"]) / denom).fillna(0.0)
    else:
        out["ob_imbalance"] = 0.0

    # Trade imbalance if trade volumes available
    if {"trades_buy_vol", "trades_sell_vol"}.issubset(out.columns):
        denom = (out["trades_buy_vol"] + out["trades_sell_vol"]).replace(0, np.nan)
        out["trade_imbalance"] = (
            (out["trades_buy_vol"] - out["trades_sell_vol"]) / denom
        ).fillna(0.0)
    else:
        out["trade_imbalance"] = 0.0

    return out


class FeatureBuilder:
    """Stateful feature builder operating on streaming bars.

    Besides returns this builder computes volatility, spread and
    microstructure indicators in a streaming fashion. A ring buffer of length
    ``seq_len`` keeps the most recent feature rows which can be retrieved via
    :meth:`window` for consumption by sequence models such as an LSTM.
    """

    def __init__(self, seq_len: int = 20, *, vol_window: int = 20) -> None:
        self.seq_len = int(seq_len)
        self.vol_window = int(vol_window)
        self._last_close: float | None = None
        self._ret_window: deque[float] = deque(maxlen=self.vol_window)
        self._buffer: deque[dict] = deque(maxlen=self.seq_len)

    def update(self, bar: dict) -> pd.DataFrame:
        close = float(bar["close"])
        ts = int(bar["timestamp"])
        ret = 0.0 if self._last_close is None else close / self._last_close - 1.0
        self._last_close = close
        self._ret_window.append(ret)
        volatility = 0.0
        if len(self._ret_window) > 1:
            volatility = float(np.std(self._ret_window, ddof=1))

        if {"bid1", "ask1"}.issubset(bar):
            bid1, ask1 = float(bar["bid1"]), float(bar["ask1"])
            spread = ask1 - bid1
            mid_price = (ask1 + bid1) / 2.0
        else:
            high = float(bar.get("high", close))
            low = float(bar.get("low", close))
            spread = high - low
            mid_price = close

        if {"bid_sz1", "ask_sz1"}.issubset(bar):
            bid_sz1 = float(bar["bid_sz1"])
            ask_sz1 = float(bar["ask_sz1"])
            denom = bid_sz1 + ask_sz1
            ob_imbalance = (bid_sz1 - ask_sz1) / denom if denom != 0 else 0.0
        else:
            ob_imbalance = 0.0

        if {"trades_buy_vol", "trades_sell_vol"}.issubset(bar):
            buy = float(bar["trades_buy_vol"])
            sell = float(bar["trades_sell_vol"])
            denom = buy + sell
            trade_imbalance = (buy - sell) / denom if denom != 0 else 0.0
        else:
            trade_imbalance = 0.0

        out = {
            "timestamp": ts,
            "ret": ret,
            "volatility": volatility,
            "spread": spread,
            "mid_price": mid_price,
            "ob_imbalance": ob_imbalance,
            "trade_imbalance": trade_imbalance,
        }
        for key, val in bar.items():
            if key not in {
                "timestamp",
                "close",
                "bid1",
                "ask1",
                "high",
                "low",
                "bid_sz1",
                "ask_sz1",
                "trades_buy_vol",
                "trades_sell_vol",
            }:
                out[key] = val
        self._buffer.append(out)
        return pd.DataFrame([out])

    def window(self) -> np.ndarray:
        """Return ``[seq_len, n_features]`` tensor of recent features."""

        if not self._buffer:
            raise ValueError("buffer is empty")
        df = pd.DataFrame(list(self._buffer))
        cols = [c for c in df.columns if c != "timestamp"]
        arr = df[cols].to_numpy(dtype=float)
        if len(arr) < self.seq_len:
            pad = np.zeros((self.seq_len - len(arr), len(cols)))
            arr = np.vstack([pad, arr])
        return arr


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


def normalize_train_only(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    columns: Sequence[str] | None = None,
    ffill_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, StandardScaler]:
    """Fit ``StandardScaler`` on ``train`` and transform both ``train`` and ``test``.

    Missing values are forward filled with a configurable limit after
    transformation.  Only the columns provided in ``columns`` are scaled.

    Returns the transformed ``train`` and ``test`` dataframes together with the
    fitted scaler instance.
    """

    cols = list(columns) if columns is not None else list(train.columns)

    # Forward fill training data prior to fitting to avoid NaNs affecting the
    # statistics. The same limit is applied as during transformation.
    train_ffill = (
        train[cols].ffill(limit=ffill_limit) if ffill_limit is not None else train[cols].ffill()
    )
    scaler = StandardScaler().fit(train_ffill)

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[cols] = (
            df[cols].ffill(limit=ffill_limit)
            if ffill_limit is not None
            else df[cols].ffill()
        )
        out[cols] = scaler.transform(out[cols])
        return out

    train_scaled = _transform(train)
    test_scaled = _transform(test) if test is not None else None
    return train_scaled, test_scaled, scaler


def sliding_window_tensor(
    df: pd.DataFrame, seq_len: int, features: Iterable[str]
) -> np.ndarray:
    """Generate ``[batch, seq_len, n_features]`` tensor using sliding windows."""

    cols = list(features)
    arr = df[cols].to_numpy(dtype=float)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    n = len(arr) - seq_len + 1
    if n <= 0:
        raise ValueError("not enough rows for the requested sequence length")
    windows = [arr[i : i + seq_len] for i in range(n)]
    return np.stack(windows)


__all__ = [
    "build_features",
    "FeatureBuilder",
    "Scaler",
    "normalize_train_only",
    "sliding_window_tensor",
]
