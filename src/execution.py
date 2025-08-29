"""Execution client interfaces for trading modes.

This module defines a minimal execution layer with a common
`ExecutionClient` interface and two concrete implementations:
`SimExecutionClient` for simulations/backtests and
`BrokerExecutionClient` as a placeholder for real broker APIs.

The goal is to decouple order generation from how orders are actually
sent or filled, enabling backtest, walk-forward and live modes to share
identical trading logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Dict, List, Optional
import uuid

from quant_pipeline.observability import Observability


# ---------------------------------------------------------------------------
# Order representation
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """Simple order container used by the engine.

    Attributes
    ----------
    symbol: str
        Instrument identifier.
    qty: float
        Quantity to trade. Positive values represent a buy and negative
        values a sell; the :class:`TradingEngine` takes care of converting
        raw signals into signed quantities.
    price: Optional[float]
        Optional limit price. ``None`` implies market order in the
        simulated client.
    id: str
        Unique order identifier generated automatically when the order is
        instantiated.
    """

    symbol: str
    qty: float
    price: Optional[float] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


# ---------------------------------------------------------------------------
# Execution client interface
# ---------------------------------------------------------------------------

class ExecutionClient(ABC):
    """Abstract base class for execution backends.

    The engine depends only on this interface making it possible to swap
    a simulated execution layer with a live broker without changing the
    decision logic.
    """

    @abstractmethod
    def send(self, order: Order) -> str:
        """Send an order to the venue.

        Returns
        -------
        str
            The unique identifier of the submitted order.
        """

    @abstractmethod
    def cancel(self, order_id: str) -> None:
        """Cancel an existing order."""

    @abstractmethod
    def positions(self) -> Dict[str, float]:
        """Return current positions keyed by symbol."""

    @abstractmethod
    def clock(self) -> datetime:
        """Return the current timestamp according to the venue."""


# ---------------------------------------------------------------------------
# Helper: simple cost model used by the simulator
# ---------------------------------------------------------------------------

class CostModel:
    """Applies linear slippage and fees in basis points."""

    def __init__(self, fee_bps: float = 0.0, slippage_bps: float = 0.0):
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

    def apply(self, price: float, qty: float) -> float:
        """Return total cost including slippage and fees."""

        slip_price = price * (1 + self.slippage_bps / 10_000)
        fee = price * abs(qty) * self.fee_bps / 10_000
        return slip_price * qty + fee


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class SimExecutionClient(ExecutionClient):
    """In-memory execution client for tests/backtests.

    Orders are assumed to fill immediately at the provided price.  A
    :class:`CostModel` instance is used to emulate fees and slippage.
    """

    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()
        self._positions: Dict[str, float] = {}
        self._ledger: List[Order] = []

    def send(self, order: Order) -> str:  # pragma: no cover - trivial
        cost = self.cost_model.apply(order.price or 0.0, order.qty)
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + order.qty
        self._ledger.append(order)
        # In a real backtest we would store PnL including cost here.
        return order.id

    def cancel(self, order_id: str) -> None:  # pragma: no cover - stub
        # In the simulation orders fill immediately, so cancel is a no-op.
        pass

    def positions(self) -> Dict[str, float]:  # pragma: no cover - trivial
        return dict(self._positions)

    def clock(self) -> datetime:  # pragma: no cover - trivial
        return datetime.now(timezone.utc)


class BrokerExecutionClient(ExecutionClient):
    """Minimal live-trading client with heartbeat and reconnect logic."""

    def __init__(self, observability: Observability | None = None, heartbeat_interval: float = 5.0):
        self._positions: Dict[str, float] = {}
        self.obs = observability or Observability()
        self._heartbeat_interval = heartbeat_interval
        self._pending_orders: Dict[str, Order] = {}
        self._connected = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.connect()
        if heartbeat_interval > 0:
            self._thread.start()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Establish connection to the broker (placeholder)."""

        self._connected = True

    def _ping(self) -> bool:  # pragma: no cover - simple stub
        """Heartbeat check to broker API."""

        return True

    def check_connection(self) -> None:
        """Verify connection and handle disconnects."""

        ok = False
        try:
            ok = self._ping()
        except Exception:
            ok = False
        if not ok:
            self._connected = False
            self.obs.alert_connection_failure("broker")
            self._cancel_all_pending()
            self.connect()
        else:
            self._connected = True

    def _run(self) -> None:
        while not self._stop.wait(self._heartbeat_interval):
            self.obs.heartbeat()
            self.check_connection()

    def close(self) -> None:
        """Stop heartbeat thread."""

        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _cancel_all_pending(self) -> None:
        for oid in list(self._pending_orders):
            self.cancel(oid)

    # ------------------------------------------------------------------
    # ExecutionClient interface
    # ------------------------------------------------------------------
    def send(self, order: Order) -> str:  # pragma: no cover - stub
        if not self._connected:
            self.connect()
        print(f"LIVE ORDER -> {order.symbol} {order.qty}@{order.price}")
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + order.qty
        self._pending_orders[order.id] = order
        self.obs.increment_orders_sent()
        return order.id

    def cancel(self, order_id: str) -> None:  # pragma: no cover - stub
        print(f"CANCEL ORDER -> {order_id}")
        self._pending_orders.pop(order_id, None)
        self.obs.increment_order_errors()

    def positions(self) -> Dict[str, float]:  # pragma: no cover - trivial
        return dict(self._positions)

    def clock(self) -> datetime:  # pragma: no cover - trivial
        return datetime.now(timezone.utc)

