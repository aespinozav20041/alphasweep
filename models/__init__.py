"""Light-weight model adapters and ensemble utilities.

This package groups small helpers used across the repository when a full
blown ML framework would be overkill.  The modules intentionally keep the
API surface minimal.
"""

from .xgb_adapter import XGBoostAdapter
from .tcn_adapter import TCNAdapter
from .ensemble import Signal, SignalEnsemble

__all__ = ["XGBoostAdapter", "TCNAdapter", "Signal", "SignalEnsemble"]
