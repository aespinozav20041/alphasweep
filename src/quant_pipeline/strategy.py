from __future__ import annotations

"""Common strategy interface."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Strategy(ABC):
    """Base class for all signal generating strategies."""

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Return model signals as a numpy array."""
        raise NotImplementedError


__all__ = ["Strategy"]
