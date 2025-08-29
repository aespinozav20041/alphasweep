"""Simple order management system with idempotent state handling."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional


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
    leverage: Optional[float] = None


class OrderWAL:
    """Simple write-ahead log for order events."""

    def __init__(self, path: Optional[str] = None):
        if path is None:
            fd, tmp = tempfile.mkstemp(prefix="oms_wal_", suffix=".log")
            os.close(fd)
            self.path = Path(tmp)
        else:
            self.path = Path(path)
        self.path.touch(exist_ok=True)

    # -- logging ---------------------------------------------------------
    def _append(self, event: Dict) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(event) + "\n")

    def record_submit(self, order: Order) -> None:
        self._append(
            {
                "type": "submit",
                "client_id": order.client_id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "price": order.price,
                "leverage": order.leverage,
            }
        )

    def record_cancel(self, client_id: str) -> None:
        self._append({"type": "cancel", "client_id": client_id})

    def record_fill(self, client_id: str, qty: float) -> None:
        self._append({"type": "fill", "client_id": client_id, "qty": qty})

    # -- replay ---------------------------------------------------------
    def replay(self) -> Iterator[Dict]:
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


class OMS:
    """Order management system ensuring reliable submissions."""

    def __init__(
        self,
        exchange,
        symbol_info: Dict[str, Dict[str, float]],
        wal_path: Optional[str] = None,
    ):
        self.exchange = exchange
        self.symbol_info = symbol_info
        self.orders: Dict[str, Order] = {}
        self.wal = OrderWAL(wal_path)

    # ------------------------------------------------------------------
    # Algorithmic scheduling
    # ------------------------------------------------------------------
    def schedule_child_orders(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        strategy: str = "twap",
        intervals: int = 1,
        volume_profile: Optional[list[float]] = None,
        participation: float = 0.1,
    ) -> list[float]:
        """Split a parent order into child slices according to strategy.

        Parameters
        ----------
        symbol, side, qty : order details
        strategy : "twap", "vwap" or "pov"
        intervals : number of slices for TWAP/VWAP
        volume_profile : expected volumes per interval for VWAP/POV
        participation : participation rate for POV
        """

        strategy = strategy.lower()
        if intervals <= 0:
            intervals = 1

        if strategy == "twap":
            slice_qty = qty / intervals
            return [slice_qty for _ in range(intervals)]
        if strategy == "vwap":
            if not volume_profile:
                volume_profile = [1.0] * intervals
            total = sum(volume_profile)
            if total == 0:
                return [qty]
            return [qty * v / total for v in volume_profile]
        if strategy == "pov":
            if not volume_profile:
                volume_profile = [1.0] * intervals
            remaining = qty
            sched: list[float] = []
            for v in volume_profile:
                slice_qty = min(remaining, participation * v)
                sched.append(slice_qty)
                remaining -= slice_qty
            if remaining > 0 and sched:
                sched[-1] += remaining
            return sched
        raise ValueError("unknown strategy")

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
        order = Order(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            client_id=client_id,
            leverage=leverage,
        )
        self.orders[client_id] = order
        self.wal.record_submit(order)
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
        self.wal.record_fill(order.client_id, qty)
        order.filled_qty += qty
        if order.filled_qty >= order.qty:
            order.state = "filled"
        else:
            order.state = "partial"

    def cancel_order(self, client_id: str) -> None:
        order = self.orders.get(client_id)
        if not order or order.state in {"filled", "canceled", "expired"}:
            return
        self.wal.record_cancel(client_id)
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
        self._send(order, leverage=order.leverage)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------
    def reconcile(self) -> None:
        """Replay WAL and load open orders from exchange."""

        # replay WAL to reconstruct internal state
        for ev in self.wal.replay():
            typ = ev.get("type")
            if typ == "submit":
                cid = ev["client_id"]
                if cid not in self.orders:
                    self.orders[cid] = Order(
                        symbol=ev["symbol"],
                        side=ev["side"],
                        qty=ev["qty"],
                        price=ev["price"],
                        client_id=cid,
                        leverage=ev.get("leverage"),
                    )
            elif typ == "cancel":
                order = self.orders.get(ev["client_id"])
                if order:
                    order.state = "canceled"
            elif typ == "fill":
                order = self.orders.get(ev["client_id"])
                if order:
                    order.filled_qty += ev["qty"]
                    if order.filled_qty >= order.qty:
                        order.state = "filled"
                    else:
                        order.state = "partial"

        # sync with exchange open orders
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

        # resend any orders not on exchange
        for order in self.orders.values():
            if order.order_id is None and order.state not in {"filled", "canceled", "expired"}:
                self._send(order, order.leverage)

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


__all__ = ["OMS", "Order", "OrderWAL", "RateLimitError"]
