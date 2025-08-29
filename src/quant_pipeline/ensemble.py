"""Simple ensemble utilities blending multiple model signals.

This module introduces a light-weight ensemble mechanism that combines the
outputs of heterogeneous models (e.g. tree-based models, neural networks and
rule based signals).  It is intentionally small but showcases how different
sources of predictive signals can be merged prior to risk management and order
slicing.

Examples
--------
>>> from sklearn.linear_model import LogisticRegression
>>> from .simple_lstm import SimpleLSTM
>>> from .ensemble import SignalEnsemble
>>> lr = LogisticRegression()
>>> lstm = SimpleLSTM(input_size=3, hidden_size=16)
>>> ens = SignalEnsemble({"lr": lr, "lstm": lstm})
>>> blended = ens.blend({"lr": 0.3, "lstm": -0.1}, weights={"lr": 0.7, "lstm": 0.3})
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


class SignalEnsemble:
    """Blend signals from multiple models.

    Parameters
    ----------
    models : mapping
        Dictionary of model objects exposing a ``predict`` method.
    default_weights : mapping, optional
        Fallback weights used when ``blend`` is called without explicit
        weights.  They will be normalised to sum to one.
    """

    def __init__(
        self,
        models: Mapping[str, object],
        *,
        default_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.models = dict(models)
        self.default_weights = (
            dict(default_weights) if default_weights is not None else None
        )

    def predict(self, name: str, X) -> np.ndarray:
        """Run inference with one of the underlying models."""

        model = self.models[name]
        if not hasattr(model, "predict"):
            raise AttributeError(f"model {name} has no predict method")
        return np.asarray(model.predict(X))

    def blend(
        self,
        signals: Mapping[str, np.ndarray | float],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Blend already-computed signals.

        ``signals`` may contain scalars or numpy arrays.  Missing models will
        raise a ``KeyError``.  ``weights`` default to ``default_weights``
        provided at construction time.  The returned blended signal is a numpy
        array whose length matches the first signal encountered.
        """

        if weights is None:
            if self.default_weights is None:
                raise ValueError("weights must be provided")
            weights = self.default_weights

        total = sum(weights.values())
        if total == 0:
            raise ValueError("weights sum to zero")

        blended = None
        for name, sig in signals.items():
            if name not in weights:
                continue
            w = weights[name] / total
            arr = np.asarray(sig)
            blended = arr * w if blended is None else blended + arr * w
        if blended is None:
            raise ValueError("no overlapping signals and weights")
        return blended


class MultiHorizonEnsemble:
    """Blend signals across multiple forecast horizons."""

    def __init__(self, ensembles: Mapping[str, SignalEnsemble]) -> None:
        self.ensembles = dict(ensembles)

    def blend(
        self,
        signals: Mapping[str, Mapping[str, np.ndarray | float]],
        *,
        weights: Mapping[str, Mapping[str, float]] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Blend signals per horizon using corresponding ensembles."""

        blended: Dict[str, np.ndarray] = {}
        for horizon, sig in signals.items():
            ens = self.ensembles[horizon]
            w = weights.get(horizon) if weights is not None else None
            blended[horizon] = ens.blend(sig, weights=w)
        return blended


__all__ = ["SignalEnsemble", "MultiHorizonEnsemble"]
