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

    trailing_pct: float = 0.0  # trailing stop expressed as fraction
    max_drawdown: float = 0.0  # hard kill-switch based on drawdown
=======
    trailing_enabled: bool = False
    atr_mult_trail: float = 3.0



class RiskManager:
    """Very small risk manager used by :class:`TradingEngine`.

    Methods ``pre_trade`` and ``post_trade`` are intentionally
    straightforward but documented so they can easily be extended with
    more sophisticated checks (drawdown, latency, etc.).
    """

    def __init__(self, limits: RiskLimits, *, trailing_on: bool = False, trailing_mult: float = 0.02):
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

        self.trailing_on = trailing_on
        self.trailing_mult = trailing_mult
        self.trailing_stop: Optional[float] = None
        self._trail_price: Optional[float] = None
        self._current_position: float = 0.0
=======
        self.trailing_enabled = limits.trailing_enabled
        self.atr_mult_trail = limits.atr_mult_trail
        # trailing stop per symbol
        self._trail: Dict[str, float] = {}
        # record of extreme prices per symbol
        self._extreme: Dict[str, float] = {}


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

    def post_trade(
        self,
        order: Order,
        positions: Dict[str, float],
        *,
        atr: Optional[float] = None,
    ) -> None:
        """Update trailing stop after a trade.

        Parameters
        ----------
        order
            Executed order information.
        positions
            Latest positions after the trade.
        atr
            Current Average True Range. If ``None`` or trailing stops are
            disabled the method is a no-op.
        """

        if not self.trailing_enabled or atr is None or atr <= 0:
            return

        pos = positions.get(order.symbol, 0.0)
        if pos == 0:
            # position flat -> remove trailing data
            self._trail.pop(order.symbol, None)
            self._extreme.pop(order.symbol, None)
            return

        price = order.price or 0.0
        extreme = self._extreme.get(order.symbol, price)

        if pos > 0:
            extreme = max(extreme, price)
            stop = extreme - self.atr_mult_trail * atr
            prev = self._trail.get(order.symbol)
            if prev is None or stop > prev:
                self._trail[order.symbol] = stop
        else:
            extreme = min(extreme, price)
            stop = extreme + self.atr_mult_trail * atr
            prev = self._trail.get(order.symbol)
            if prev is None or stop < prev:
                self._trail[order.symbol] = stop

        self._extreme[order.symbol] = extreme

    # ------------------------------------------------------------------
    def limit_order(self, order: Order) -> Order:
        """Return order with price constrained by trailing stop.

        If a trailing stop exists for ``order.symbol`` the limit price is
        adjusted so that it cannot cross the trailing level.
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
=======

        if self.trailing_on:
            current = positions.get(order.symbol, 0.0)
            if current == 0:
                self.trailing_stop = None
                self._trail_price = None
        return
=======
        if not self.trailing_enabled:
            return order
        stop = self._trail.get(order.symbol)
        if stop is None:
            return order
        if order.qty < 0:
            if order.price is None or order.price < stop:
                order.price = stop
        elif order.qty > 0:
            if order.price is None or order.price > stop:
                order.price = stop
        return order


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


