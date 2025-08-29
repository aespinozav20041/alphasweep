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
    trailing_pct: float = 0.0  # trailing stop expressed as fraction
    max_drawdown: float = 0.0  # hard kill-switch based on drawdown


class RiskManager:
    """Very small risk manager used by :class:`TradingEngine`.

    Methods ``pre_trade`` and ``post_trade`` are intentionally
    straightforward but documented so they can easily be extended with
    more sophisticated checks (drawdown, latency, etc.).
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self._entry: Dict[str, float] = {}
        self._high_water: Dict[str, float] = {}
        self._kill = False

    def update_price(self, symbol: str, price: float) -> None:
        """Update trailing stop and drawdown checks with latest price."""

        if symbol in self._entry:
            self._high_water[symbol] = max(self._high_water.get(symbol, price), price)
            if self.limits.trailing_pct > 0:
                stop = self._high_water[symbol] * (1 - self.limits.trailing_pct)
                if price <= stop:
                    self._kill = True
            if self.limits.max_drawdown > 0:
                drawdown = (
                    self._high_water[symbol] - price
                ) / self._high_water[symbol]
                if drawdown >= self.limits.max_drawdown:
                    self._kill = True

    # ------------------------------------------------------------------
    # Hooks called by the engine
    # ------------------------------------------------------------------
    def pre_trade(self, order: Order, positions: Dict[str, float]) -> bool:
        """Return ``True`` if the order respects risk limits."""

        if self._kill:
            return False
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

        current = positions.get(order.symbol, 0.0)
        if current == 0:
            self._entry.pop(order.symbol, None)
            self._high_water.pop(order.symbol, None)
        else:
            price = order.price or 0.0
            if order.symbol not in self._entry:
                self._entry[order.symbol] = price
                self._high_water[order.symbol] = price

    def reset_kill(self) -> None:
        """Reset kill-switch allowing trading again."""

        self._kill = False

