"""Unified trading engine used by all modes.

The :class:`TradingEngine` glues together feature construction, model
prediction, position sizing, risk management and order execution.  The
same engine is used by backtests, walk-forward experiments and live
trading simply by swapping the :class:`ExecutionClient` instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from execution import ExecutionClient, Order
from risk import RiskManager


@dataclass
class TradingEngine:
    """Minimal yet extensible trading engine."""

    model: Any
    feature_builder: Callable[[Any], Any]
    risk_manager: RiskManager
    execution_client: ExecutionClient
    size_fn: Optional[Callable[[float], float]] = None

    def __post_init__(self) -> None:
        # Default position sizing: buy one unit per unit signal
        if self.size_fn is None:
            self.size_fn = lambda s: 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)

    # ------------------------------------------------------------------
    def on_bar(self, market_data: Any) -> Optional[str]:
        """Process one market data event and optionally send an order."""

        self.risk_manager.update_price(
            getattr(market_data, "symbol", ""), getattr(market_data, "price", 0.0)
        )
        features = self.feature_builder(market_data)
        signal = float(self.model.predict(features))
        qty = self.size_fn(signal)
        if qty == 0:
            return None
        side_price = getattr(market_data, "price", None)
        order = Order(symbol=getattr(market_data, "symbol", ""), qty=qty, price=side_price)
        if not self.risk_manager.pre_trade(order, self.execution_client.positions()):
            return None
        order_id = self.execution_client.send(order)
        self.risk_manager.post_trade(order, self.execution_client.positions())
        return order_id

    # Convenience aliases ------------------------------------------------
    def positions(self):  # pragma: no cover - simple pass-through
        return self.execution_client.positions()

    def clock(self):  # pragma: no cover - simple pass-through
        return self.execution_client.clock()

