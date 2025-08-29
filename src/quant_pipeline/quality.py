"""Market data quality checks for streaming bars."""

from __future__ import annotations

import time
from typing import Dict

# Minimal fields required for downstream components
REQUIRED_FIELDS = {"timestamp", "symbol", "close"}

def quality_check(bar: Dict[str, float], *, max_delay: float = 60.0) -> bool:
    """Validate presence of required fields and freshness of a bar.

    Parameters
    ----------
    bar:
        Market data tick containing OHLCV information.
    max_delay:
        Maximum allowed age of the bar in seconds.

    Returns
    -------
    bool
        ``True`` if the bar passes all quality checks, ``False`` otherwise.
    """
    if not REQUIRED_FIELDS.issubset(bar):
        return False
    try:
        ts = float(bar["timestamp"])
    except (TypeError, ValueError):
        return False
    # Normalize to seconds if timestamp provided in milliseconds
    if ts > 1e12:
        ts /= 1000.0
    # Only enforce freshness if timestamp resembles a UNIX epoch value
    if ts > 1e9 and time.time() - ts > max_delay:
        return False
    return True


__all__ = ["quality_check"]
