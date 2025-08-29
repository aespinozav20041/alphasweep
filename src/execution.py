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
from typing import Dict, List, Optional
import uuid


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
    """Placeholder implementation for live trading.

    A :class:`CostModel` is used to record fees and slippage so that
    live trading can track realised costs similar to the simulator.  The
    client keeps a ledger of submitted orders which are cancelled upon
    disconnect.
    """

    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()
        self._positions: Dict[str, float] = {}
        self._orders: Dict[str, Order] = {}
        self._ledger: List[Dict[str, float]] = []

    def send(self, order: Order) -> str:  # pragma: no cover - stub
        cost = self.cost_model.apply(order.price or 0.0, order.qty)
        print(f"LIVE ORDER -> {order.symbol} {order.qty}@{order.price} cost={cost:.4f}")
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + order.qty
        self._orders[order.id] = order
        self._ledger.append({"id": order.id, "cost": cost})
        return order.id

    def cancel(self, order_id: str) -> None:  # pragma: no cover - stub
        print(f"CANCEL ORDER -> {order_id}")
        self._orders.pop(order_id, None)

    def cancel_all(self) -> None:  # pragma: no cover - stub
        for oid in list(self._orders.keys()):
            self.cancel(oid)

    def handle_disconnect(self) -> None:  # pragma: no cover - stub
        """Cancel all resting orders on a disconnect event."""

        self.cancel_all()

    def positions(self) -> Dict[str, float]:  # pragma: no cover - trivial
        return dict(self._positions)

    def clock(self) -> datetime:  # pragma: no cover - trivial
        return datetime.now(timezone.utc)

