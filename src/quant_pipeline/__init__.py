"""Core package for quant-pipeline."""

from .strategy import Strategy
from .simple_lstm import SimpleLSTM
from .moving_average import MovingAverageStrategy

__all__ = ["__version__", "Strategy", "SimpleLSTM", "MovingAverageStrategy"]
__version__ = "0.1.0"
