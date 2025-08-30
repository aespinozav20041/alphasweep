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

from quant_pipeline.storage import record_prediction
=======
from trading import sor



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
=======


        features = self.feature_builder(market_data)
        signal = float(self.model.predict(features))

        ts = getattr(market_data, "timestamp", int(self.execution_client.clock().timestamp() * 1000))
        symbol = getattr(market_data, "symbol", "")
        p_long = p_short = 0.0
        prob_fn = getattr(self.model, "predict_proba", None)
        if prob_fn is not None:
            try:
                probs = prob_fn(features)
                if isinstance(probs, (list, tuple)) and len(probs) >= 2:
                    p_short = float(probs[0])
                    p_long = float(probs[1])
            except Exception:
                pass
        record_prediction(symbol, ts, signal, p_long, p_short)

=======
        bar = self.feature_builder(market_data)
        symbol = getattr(market_data, "symbol", "")
        signal = float(self.model.predict(bar, symbol=symbol))
        qty = self.size_fn(signal)
        if qty == 0:
            return None
        side_price = getattr(market_data, "price", None)
        order = Order(
            symbol=getattr(market_data, "symbol", ""),
            qty=qty,
            price=side_price,
            spread=getattr(market_data, "spread", 0.0),
            vol=getattr(market_data, "volatility", 0.0),
            volume=getattr(market_data, "volume", 0.0),
        )
        if not self.risk_manager.pre_trade(order, self.execution_client.positions()):
            return None
        order_id = sor.route_order(order, self.execution_client)
        self.risk_manager.post_trade(order, self.execution_client.positions())
=======
        order = self.risk_manager.limit_order(order)
        order_id = self.execution_client.send(order)
        atr = getattr(market_data, "atr", None)
        self.risk_manager.post_trade(order, self.execution_client.positions(), atr=atr)

        return order_id

    # Convenience aliases ------------------------------------------------
    def positions(self):  # pragma: no cover - simple pass-through
        return self.execution_client.positions()

    def clock(self):  # pragma: no cover - simple pass-through
        return self.execution_client.clock()

