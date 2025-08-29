"""Basic risk management utilities.

The real project contains a fairly involved risk module.  This simplified
version keeps only the pieces required by the example trading engine so
that backtest, walk-forward and live modes can share the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from execution import Order


@dataclass
class RiskLimits:
    """Configuration for simple notional limits."""

    max_position: float = 0.0  # absolute quantity limit per symbol


class RiskManager:
    """Very small risk manager used by :class:`TradingEngine`.

    Methods ``pre_trade`` and ``post_trade`` are intentionally
    straightforward but documented so they can easily be extended with
    more sophisticated checks (drawdown, latency, etc.).
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits

    # ------------------------------------------------------------------
    # Hooks called by the engine
    # ------------------------------------------------------------------
    def pre_trade(self, order: Order, positions: Dict[str, float]) -> bool:
        """Return ``True`` if the order respects position limits."""

        current = positions.get(order.symbol, 0.0)
        proposed = current + order.qty
        if abs(proposed) > self.limits.max_position:
            return False
        return True

    def post_trade(self, order: Order, positions: Dict[str, float]) -> None:
        """Placeholder post-trade hook.

        In a full featured implementation this method would update
        drawdown statistics, latency measurements, etc.  The current
        version simply exists so the engine has a stable callback point.
        """

        # No-op for now, reserved for future extensions.
        return

