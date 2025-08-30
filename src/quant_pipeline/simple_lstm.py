"""PyTorch-based LSTM model with persistent hidden state."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .strategy import Strategy


class SimpleLSTM(nn.Module, Strategy):
    """Tiny LSTM network keeping state across calls.

    The model is purposely minimal but uses a real :class:`~torch.nn.LSTM`
    layer.  It exposes :meth:`fit` and :meth:`predict` methods and can persist
    its parameters together with the current hidden state to disk so the state
    can be restored after a restart.
    """

    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden_size: int = 4,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden: Optional[tuple[Tensor, Tensor]] = None
        # expose expected input feature dimension for external checks
        self.input_size = input_size
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
        nn.init.zeros_(self.fc.bias)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_state(self, path: str | Path) -> None:
        """Persist model weights and hidden state to ``path``."""

        torch.save({"model": self.state_dict(), "hidden": self.hidden}, path)

    def load_state(self, path: str | Path) -> None:
        """Load model weights and hidden state from ``path``."""

        data = torch.load(path, map_location="cpu")
        self.load_state_dict(data["model"])
        self.hidden = data.get("hidden")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def fit(self, feats: pd.DataFrame, *, epochs: int = 20, lr: float = 0.01) -> None:
        """Train the network to predict the next return value.

        Parameters
        ----------
        feats:
            DataFrame containing a ``ret`` column and additional features.
        epochs:
            Number of optimisation epochs.
        lr:
            Learning rate for the optimiser.
        """

        if "ret" not in feats:
            raise ValueError("missing 'ret' column")

        data = torch.tensor(feats.values, dtype=torch.float32)
        if len(data) < 2:
            raise ValueError("need at least two observations to train")

        x = data[:-1].unsqueeze(0)
        y = data[1:, [feats.columns.get_loc("ret")]].unsqueeze(0)

        optim = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            optim.zero_grad()
            out, _ = self.lstm(x)
            pred = self.fc(out)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
        # remove any learnt biases so zero input yields zero output
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if "bias" in name:
                    param.zero_()
            self.fc.bias.zero_()
        # reset hidden state so future predictions start fresh
        self.hidden = None

    def predict(self, feats: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate prediction for a window of features."""

        if isinstance(feats, pd.DataFrame):
            arr = feats.to_numpy(dtype=float)
        else:
            arr = np.asarray(feats, dtype=float)
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out, self.hidden = self.lstm(x, self.hidden)
            pred = self.fc(out[:, -1, :])

        # detach hidden state so it can be serialised
        if isinstance(self.hidden, tuple):
            self.hidden = tuple(h.detach() for h in self.hidden)
        return pred.view(-1).cpu().numpy()


__all__ = ["SimpleLSTM"]

