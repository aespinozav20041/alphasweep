from __future__ import annotations
"""Template adapter for Temporal Convolutional Networks."""

from typing import Any


class TCNAdapter:
    """Minimal TCN wrapper exposing ``predict``."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, features: Any) -> float:
        return float(self.model(features))
