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

from quant_pipeline.storage import record_fill, record_equity


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
        self._cash: float = 0.0
        self._last_price: Dict[str, float] = {}

    def send(self, order: Order) -> str:  # pragma: no cover - trivial
        price = order.price or 0.0
        slip_price = price * (1 + self.cost_model.slippage_bps / 10_000)
        fee = price * abs(order.qty) * self.cost_model.fee_bps / 10_000
        slippage_amt = (slip_price - price) * order.qty
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + order.qty
        self._ledger.append(order)
        self._last_price[order.symbol] = slip_price
        self._cash -= slip_price * order.qty + fee

        ts = int(self.clock().timestamp() * 1000)
        record_fill(order.id, ts, slip_price, order.qty, fee, slippage_amt)
        nav = self._cash + sum(
            pos * self._last_price.get(sym, 0.0)
            for sym, pos in self._positions.items()
        )
        exposure = sum(
            abs(pos * self._last_price.get(sym, 0.0))
            for sym, pos in self._positions.items()
        )
        record_equity(ts, nav, self._cash, exposure)

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

    In a production system this class would wrap a library such as CCXT,
    IBKR, or a broker SDK.  The methods currently log their usage making
    it easy to later plug in the real API calls.
    """

    def __init__(self):
        self._positions: Dict[str, float] = {}
        self._cash: float = 0.0
        self._last_price: Dict[str, float] = {}

    def send(self, order: Order) -> str:  # pragma: no cover - stub
        # Replace the print statements with real broker API calls.
        price = order.price or 0.0
        print(f"LIVE ORDER -> {order.symbol} {order.qty}@{price}")
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + order.qty
        self._last_price[order.symbol] = price
        self._cash -= price * order.qty

        ts = int(self.clock().timestamp() * 1000)
        record_fill(order.id, ts, price, order.qty, 0.0, 0.0)
        nav = self._cash + sum(
            pos * self._last_price.get(sym, 0.0)
            for sym, pos in self._positions.items()
        )
        exposure = sum(
            abs(pos * self._last_price.get(sym, 0.0))
            for sym, pos in self._positions.items()
        )
        record_equity(ts, nav, self._cash, exposure)

        return order.id

    def cancel(self, order_id: str) -> None:  # pragma: no cover - stub
        print(f"CANCEL ORDER -> {order_id}")

    def positions(self) -> Dict[str, float]:  # pragma: no cover - trivial
        return dict(self._positions)

    def clock(self) -> datetime:  # pragma: no cover - trivial
        return datetime.now(timezone.utc)

