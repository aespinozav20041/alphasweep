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


def blend_horizons(signals: Dict[int, np.ndarray | float]) -> np.ndarray:
    """Blend signals coming from different forecast horizons.

    Each entry in ``signals`` maps a horizon (in bars) to the signal produced
    for that horizon.  The horizons are combined using weights inversely
    proportional to their length so that nearer horizons carry more influence.

    Parameters
    ----------
    signals
        Mapping from horizon to already computed signal values.  Values may be
        scalars or NumPy arrays and must all share the same shape.

    Returns
    -------
    numpy.ndarray
        Weighted blend of the provided signals.
    """

    if not signals:
        raise ValueError("signals must not be empty")

    weights = {h: 1.0 / float(h) for h in signals}
    total = sum(weights.values())
    blended = None
    for h, sig in signals.items():
        arr = np.asarray(sig, dtype=float)
        w = weights[h] / total
        blended = arr * w if blended is None else blended + arr * w
    return blended

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

    def blend_multi(
        self,
        signals: Mapping[str, Mapping[str, Mapping[str, np.ndarray | float]]],
        *,
        model_weights: Mapping[str, float],
        horizon_weights: Mapping[str, float],
    ) -> Dict[str, np.ndarray]:
        """Blend signals across symbols and horizons."""

        out: Dict[str, np.ndarray] = {}
        for symbol, horizons in signals.items():
            blended_h: np.ndarray | None = None
            total_w = 0.0
            for horizon, sigs in horizons.items():
                if horizon not in horizon_weights:
                    continue
                model_blend = self.blend(sigs, weights=model_weights)
                w = horizon_weights[horizon]
                total_w += w
                blended_h = (
                    model_blend * w
                    if blended_h is None
                    else blended_h + model_blend * w
                )
            if blended_h is not None and total_w > 0:
                out[symbol] = blended_h / total_w
        if not out:
            raise ValueError("no blended signals produced")
        return out


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



__all__ = ["SignalEnsemble", "blend_horizons"]
=======
__all__ = ["SignalEnsemble", "MultiHorizonEnsemble"]

