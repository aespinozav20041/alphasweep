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
# Helper structures and cost model used by the simulator
# ---------------------------------------------------------------------------


@dataclass
class FeeSchedule:
    """Maker/taker fee schedule per exchange and tier in basis points."""

    fees: Dict[str, Dict[str, Dict[str, float]]]

    def get(self, exchange: str, tier: str, side: str) -> float:
        """Return the fee (in bps) for the given exchange, tier and side."""

        return self.fees.get(exchange, {}).get(tier, {}).get(side, 0.0)


class CostModel:
    """Applies linear slippage and fees in basis points."""

    def __init__(
        self,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        fee_schedule: Optional[FeeSchedule] = None,
    ):
        # ``fee_bps`` provides a simple flat fee for backwards compatibility
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.fee_schedule = fee_schedule

    def apply(
        self,
        price: float,
        qty: float,
        side: str = "taker",
        exchange: str = "default",
        tier: str = "0",
    ) -> float:
        """Return total cost including slippage and fees.

        Parameters
        ----------
        price:
            Fill price of the order.
        qty:
            Executed quantity. Positive for buys, negative for sells.
        side:
            Either ``"maker"`` or ``"taker"``; selects the fee from the
            schedule.  Defaults to ``"taker"``.
        exchange:
            Exchange identifier used in the fee schedule. Defaults to
            ``"default"``.
        tier:
            Volume tier used in the fee schedule. Defaults to ``"0"``.
        """

        slip_price = price * (1 + self.slippage_bps / 10_000)

        fee_bps = self.fee_bps
        if self.fee_schedule is not None:
            fee_bps = self.fee_schedule.get(exchange, tier, side)

        fee = price * abs(qty) * fee_bps / 10_000
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

    In a production system this class would wrap a library such as CCXT,
    IBKR, or a broker SDK.  The methods currently log their usage making
    it easy to later plug in the real API calls.
    """

    def __init__(self):
        self._positions: Dict[str, float] = {}

    def send(self, order: Order) -> str:  # pragma: no cover - stub
        # Replace the print statements with real broker API calls.
        print(f"LIVE ORDER -> {order.symbol} {order.qty}@{order.price}")
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + order.qty
        return order.id

    def cancel(self, order_id: str) -> None:  # pragma: no cover - stub
        print(f"CANCEL ORDER -> {order_id}")

    def positions(self) -> Dict[str, float]:  # pragma: no cover - trivial
        return dict(self._positions)

    def clock(self) -> datetime:  # pragma: no cover - trivial
        return datetime.now(timezone.utc)

