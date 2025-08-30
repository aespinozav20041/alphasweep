"""Adapter wrapping XGBoost models.

The adapter exposes a very small API mirroring the parts of the scikit-learn
interface that are required within the project.  Predictions are returned as
:class:`~models.ensemble.Signal` objects so they can be fed directly into
:class:`~models.ensemble.SignalEnsemble`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - handled gracefully
    XGBRegressor = None  # type: ignore

from .ensemble import Signal


class XGBoostAdapter:
    """Light-weight wrapper around :class:`xgboost.XGBRegressor`."""

    def __init__(self, **model_kwargs: Any) -> None:
        if XGBRegressor is None:  # pragma: no cover - only triggers if missing
            raise ImportError("xgboost is required for XGBoostAdapter")
        self.model = XGBRegressor(**model_kwargs)

    def fit(
        self,
        X: np.ndarray | Sequence[Sequence[float]],
        y: Sequence[float],
        **kwargs: Any,
    ) -> None:
        """Fit the underlying model."""

        self.model.fit(np.asarray(X), np.asarray(y), **kwargs)

    def predict(self, X: np.ndarray | Sequence[Sequence[float]]) -> list[Signal]:
        """Generate :class:`Signal` objects for feature matrix ``X``."""

        preds = self.model.predict(np.asarray(X))
        return [Signal(float(p)) for p in np.atleast_1d(preds)]


__all__ = ["XGBoostAdapter"]
