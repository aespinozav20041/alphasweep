"""Basic risk management utilities.

The real project contains a fairly involved risk module.  This simplified
version keeps only the pieces required by the example trading engine so
that backtest, walk-forward and live modes can share the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

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

    def __init__(self, limits: RiskLimits, *, trailing_on: bool = False, trailing_mult: float = 0.02):
        self.limits = limits
        self.trailing_on = trailing_on
        self.trailing_mult = trailing_mult
        self.trailing_stop: Optional[float] = None
        self._trail_price: Optional[float] = None
        self._current_position: float = 0.0

    # ------------------------------------------------------------------
    # Hooks called by the engine
    # ------------------------------------------------------------------
    def pre_trade(self, order: Order, positions: Dict[str, float]) -> bool:
        """Return ``True`` if the order respects position limits."""

        current = positions.get(order.symbol, 0.0)
        proposed = current + order.qty
        if abs(proposed) > self.limits.max_position:
            return False

        if self.trailing_on and order.price is not None:
            self._current_position = current
            if current == 0:
                self.trailing_stop = None
                self._trail_price = None
            else:
                self.update_trailing(order.price)
                if self.trailing_stop is not None:
                    hit_long = current > 0 and order.price <= self.trailing_stop and order.qty > 0
                    hit_short = current < 0 and order.price >= self.trailing_stop and order.qty < 0
                    if hit_long or hit_short:
                        return False

        return True

    def post_trade(self, order: Order, positions: Dict[str, float]) -> None:
        """Placeholder post-trade hook.

        In a full featured implementation this method would update
        drawdown statistics, latency measurements, etc.  The current
        version simply exists so the engine has a stable callback point.
        """

        if self.trailing_on:
            current = positions.get(order.symbol, 0.0)
            if current == 0:
                self.trailing_stop = None
                self._trail_price = None
        return

    def update_trailing(self, price: float) -> float | None:
        """Update trailing stop based on a favorable move."""

        if not self.trailing_on or self._current_position == 0:
            return self.trailing_stop

        if self._current_position > 0:
            if self._trail_price is None or price > self._trail_price:
                self._trail_price = price
            new_stop = self._trail_price * (1 - self.trailing_mult)
            if self.trailing_stop is None or new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
        else:
            if self._trail_price is None or price < self._trail_price:
                self._trail_price = price
            new_stop = self._trail_price * (1 + self.trailing_mult)
            if self.trailing_stop is None or new_stop < self.trailing_stop:
                self.trailing_stop = new_stop
        return self.trailing_stop

