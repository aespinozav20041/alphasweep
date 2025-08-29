"""Model adapters and rule-based strategies for diverse signal sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import torch
from torch import nn


class XGBAdapter:
    """Light-weight adapter wrapping a gradient boosting regressor.

    The project does not depend on the ``xgboost`` package to keep the
    footprint small.  Instead we approximate the behaviour using
    :class:`sklearn.ensemble.GradientBoostingRegressor` which provides a
    compatible ``fit``/``predict`` API for tabular features.
    """

    def __init__(self, **params) -> None:
        self.model = GradientBoostingRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class TCNAdapter(nn.Module):
    """Very small Temporal Convolutional Network for sequence features."""

    def __init__(self, input_size: int, channels: int = 8, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1)
        self.conv = nn.Conv1d(input_size, channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` expected shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # to (batch, features, seq)
        y = self.relu(self.conv(x))
        return self.fc(y[:, :, -1])

    def fit(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 10, lr: float = 1e-3) -> None:
        self.train()
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        x_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        for _ in range(epochs):
            optim.zero_grad()
            pred = self.forward(x_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optim.step()
        self.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            pred = self.forward(x_t)
        return pred.view(-1).numpy()


@dataclass
class RuleStrategy:
    """Simple rule-based strategy using threshold on feature mean."""

    threshold: float = 0.0

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            arr = features.to_numpy(dtype=float)
        else:
            arr = np.asarray(features, dtype=float)
        signal = np.where(arr.mean(axis=1) > self.threshold, 1.0, -1.0)
        return signal


__all__ = ["XGBAdapter", "TCNAdapter", "RuleStrategy"]
