from __future__ import annotations

"""Simple moving average crossover strategy."""

import numpy as np
import pandas as pd

from .strategy import Strategy


class MovingAverageStrategy(Strategy):
    """Generates +1/-1 signal based on moving average crossover."""

    def __init__(self, short_window: int = 3, long_window: int = 5) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("windows must be positive")
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.short_window = short_window
        self.long_window = long_window

    def predict(self, prices: pd.Series | np.ndarray) -> np.ndarray:
        arr = np.asarray(prices, dtype=float)
        if arr.ndim != 1 or len(arr) < self.long_window:
            raise ValueError("not enough data for moving averages")
        s = pd.Series(arr)
        short = s.rolling(self.short_window).mean()
        long = s.rolling(self.long_window).mean()
        signal = np.where(short > long, 1.0, -1.0)
        return signal[self.long_window - 1 :]


__all__ = ["MovingAverageStrategy"]
