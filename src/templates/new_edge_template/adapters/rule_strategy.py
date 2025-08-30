from __future__ import annotations
"""Example rule based strategy template."""

from typing import Any


class RuleStrategy:
    """Simple moving-average crossover example."""

    def __init__(self, fast: int = 5, slow: int = 20) -> None:
        self.fast = fast
        self.slow = slow

    def predict(self, prices: Any) -> float:
        fast_ma = prices[-self.fast :].mean()
        slow_ma = prices[-self.slow :].mean()
        return float(fast_ma > slow_ma) - float(fast_ma < slow_ma)
