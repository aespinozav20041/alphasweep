from __future__ import annotations
"""Template adapter for XGBoost models."""

from typing import Any


class XGBAdapter:
    """Example wrapper exposing a unified ``predict`` API."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, features: Any) -> float:
        return float(self.model.predict(features))
