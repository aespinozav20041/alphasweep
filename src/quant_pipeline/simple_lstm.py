"""Minimal stateful LSTM-like model used for tests."""

from __future__ import annotations

import pandas as pd


class SimpleLSTM:
    """Very small LSTM-style model maintaining an internal state.

    The model is intentionally lightweight and only exposes a :meth:`predict`
    method used by the decision loop. It keeps an internal hidden state and
    performs a simple exponential smoothing of the input feature.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.h = 0.0

    def predict(self, feats: pd.DataFrame) -> list[float]:
        """Generate a prediction given a dataframe of features.

        Parameters
        ----------
        feats: DataFrame
            Must contain a ``ret`` column. Only the last row is considered.
        """

        val = float(feats.iloc[-1]["ret"])
        self.h = self.alpha * val + (1 - self.alpha) * self.h
        return [self.h]


__all__ = ["SimpleLSTM"]
