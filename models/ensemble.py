"""Tools for combining model signals.

This module defines :class:`Signal`, a small data container describing a
model output together with its associated quality metrics, and
:class:`SignalEnsemble` which aggregates signals from different horizons
using a weighted average based on confidence and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass
class Signal:
    """Representation of a model prediction.

    Parameters
    ----------
    value:
        Predicted value or signal strength.
    confidence:
        Value between 0 and 1 expressing how confident the model is in the
        prediction.  Defaults to ``1.0``.
    calibration:
        Calibration score of the model.  Defaults to ``1.0``.
    """

    value: float
    confidence: float = 1.0
    calibration: float = 1.0

    @property
    def weight(self) -> float:
        """Weight contribution derived from confidence and calibration."""

        return self.confidence * self.calibration


class SignalEnsemble:
    """Combine signals from multiple time horizons.

    The ensemble expects a mapping containing the horizons ``h1m``, ``h5m``
    and ``h60m`` by default.  Each value should be a :class:`Signal` instance.
    The combined output is a weighted average where weights are the product
    of each signal's confidence and calibration.
    """

    def __init__(self, horizons: Iterable[str] | None = None) -> None:
        self.horizons = list(horizons) if horizons is not None else ["h1m", "h5m", "h60m"]

    def combine(self, signals: Mapping[str, Signal]) -> float:
        """Blend the provided signals into a single value."""

        weights = []
        values = []
        for h in self.horizons:
            if h not in signals:
                raise KeyError(f"missing signal for horizon {h}")
            sig = signals[h]
            w = sig.weight
            weights.append(w)
            values.append(sig.value)
        total = float(np.sum(weights))
        if total == 0:
            raise ValueError("total weight is zero")
        return float(np.dot(values, weights) / total)


__all__ = ["Signal", "SignalEnsemble"]
