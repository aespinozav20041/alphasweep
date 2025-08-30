"""Temporal Convolutional Network (TCN) adapter.

The implementation is intentionally compact and focuses on exposing an easy to
use ``fit``/``predict`` interface.  It relies only on PyTorch and is suitable
for tiny experimental datasets rather than production use.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from .ensemble import Signal


class _SimpleTCN(nn.Module):
    """Very small TCN made of two causal convolution layers."""

    def __init__(self, input_size: int, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = (kernel_size - 1)
        self.conv1 = nn.Conv1d(
            input_size, channels, kernel_size, padding=padding, dilation=1
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=2
        )
        self.fc = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.fc(out[:, :, -1])


class TCNAdapter(nn.Module):
    """Adapter exposing ``fit`` and ``predict`` for a tiny TCN."""

    def __init__(
        self,
        *,
        input_size: int = 1,
        channels: int = 8,
        kernel_size: int = 2,
    ) -> None:
        super().__init__()
        self.model = _SimpleTCN(input_size, channels, kernel_size)
        self.input_size = input_size

    def fit(
        self,
        X: np.ndarray | Sequence[Sequence[float]],
        y: Sequence[float],
        *,
        epochs: int = 20,
        lr: float = 0.001,
    ) -> None:
        """Train the network using mean squared error."""

        x_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim != 3 or x_arr.shape[2] != self.input_size:
            raise ValueError("X must have shape (n_samples, seq_len, input_size)")
        x = torch.tensor(x_arr, dtype=torch.float32).transpose(1, 2)
        t = torch.tensor(y_arr, dtype=torch.float32).view(-1, 1)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            opt.zero_grad()
            pred = self.model(x)
            loss = loss_fn(pred, t)
            loss.backward()
            opt.step()

    def predict(self, X: np.ndarray | Sequence[Sequence[float]]) -> list[Signal]:
        """Return predictions as :class:`Signal` objects."""

        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim != 3 or x_arr.shape[2] != self.input_size:
            raise ValueError("X must have shape (n_samples, seq_len, input_size)")
        x = torch.tensor(x_arr, dtype=torch.float32).transpose(1, 2)
        with torch.no_grad():
            pred = self.model(x).view(-1).tolist()
        return [Signal(float(p)) for p in pred]


__all__ = ["TCNAdapter"]
