"""Simple order management system with idempotent state handling."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional


logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Exception raised when exchange rate limit is hit."""


@dataclass
class Order:
    """Internal order representation."""

    symbol: str
    side: str
    qty: float
    price: float
    client_id: str
    state: str = "new"
    filled_qty: float = 0.0
    order_id: Optional[str] = None


class OMS:
    """Order management system ensuring reliable submissions."""

    def __init__(self, exchange, symbol_info: Dict[str, Dict[str, float]]):
        self.exchange = exchange
        self.symbol_info = symbol_info
        self.orders: Dict[str, Order] = {}

    # ------------------------------------------------------------------
    # Order submission and idempotency
    # ------------------------------------------------------------------
    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        client_id: str,
        leverage: Optional[float] = None,
    ) -> Order:
        """Validate and submit an order to the exchange.

        If ``client_id`` already exists the existing order is returned to
        guarantee idempotency.
        """

        if client_id in self.orders:
            logger.info("idempotent submission for %s", client_id)
            return self.orders[client_id]
        if not self._validate(symbol, price, qty, leverage):
            raise ValueError("order parameters violate symbol constraints")
        order = Order(symbol=symbol, side=side, qty=qty, price=price, client_id=client_id)
        self.orders[client_id] = order
        self._send(order, leverage)
        return order

    def _send(self, order: Order, leverage: Optional[float]) -> None:
        tries, delay = 3, 0.1
        for _ in range(tries):
            try:
                order.order_id = self.exchange.create_order(
                    order.symbol,
                    order.side,
                    order.qty,
                    order.price,
                    order.client_id,
                    leverage=leverage,
                )
                order.state = "working"
                return
            except RateLimitError as exc:
                logger.warning("rate limited: %s", exc)
                time.sleep(delay)
                delay *= 2
        raise RateLimitError("create_order failed after retries")

    # ------------------------------------------------------------------
    # Order state updates
    # ------------------------------------------------------------------
    def handle_fill(self, order_id: str, qty: float) -> None:
        """Process fill event from exchange."""

        order = self._by_order_id(order_id)
        if not order:
            return
        order.filled_qty += qty
        if order.filled_qty >= order.qty:
            order.state = "filled"
        else:
            order.state = "partial"

    def cancel_order(self, client_id: str) -> None:
        order = self.orders.get(client_id)
        if not order or order.state in {"filled", "canceled", "expired"}:
            return
        if order.order_id:
            self.exchange.cancel_order(order.order_id)
        order.state = "canceled"

    def replace_order(self, client_id: str, *, price: Optional[float] = None, qty: Optional[float] = None) -> None:
        order = self.orders.get(client_id)
        if not order:
            raise KeyError(client_id)
        if order.state in {"filled", "canceled", "expired"}:
            raise ValueError("cannot replace finalized order")
        if order.order_id:
            self.exchange.cancel_order(order.order_id)
        order.state = "canceled"
        if price is not None:
            order.price = price
        if qty is not None:
            order.qty = qty
            order.filled_qty = 0.0
        order.state = "new"
        self._send(order, leverage=None)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------
    def reconcile(self) -> None:
        """Load open orders from exchange without duplicating existing ones."""

        for info in self.exchange.get_open_orders():
            cid = info["client_id"]
            order = self.orders.get(cid)
            if order is None:
                order = Order(
                    symbol=info["symbol"],
                    side=info["side"],
                    qty=info["qty"],
                    price=info["price"],
                    client_id=cid,
                    order_id=info["order_id"],
                    filled_qty=info.get("filled_qty", 0.0),
                    state="partial" if info.get("filled_qty") else "working",
                )
                self.orders[cid] = order
            else:
                order.order_id = info["order_id"]
                order.filled_qty = info.get("filled_qty", order.filled_qty)
                if order.filled_qty >= order.qty:
                    order.state = "filled"
                elif order.filled_qty > 0:
                    order.state = "partial"
                else:
                    order.state = "working"

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate(self, symbol: str, price: float, qty: float, leverage: Optional[float]) -> bool:
        info = self.symbol_info.get(symbol, {})
        tick = info.get("tick_size", 0)
        lot = info.get("lot_size", 0)
        min_notional = info.get("min_notional", 0)
        max_leverage = info.get("max_leverage")
        if tick and round(price / tick) * tick != price:
            return False
        if lot and round(qty / lot) * lot != qty:
            return False
        if price * qty < min_notional:
            return False
        if max_leverage is not None and leverage is not None and leverage > max_leverage:
            return False
        return True

    def _by_order_id(self, order_id: str) -> Optional[Order]:
        for order in self.orders.values():
            if order.order_id == order_id:
                return order
        return None


__all__ = ["OMS", "Order", "RateLimitError"]
