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
from typing import Callable, Dict, List, Optional
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
    spread: float
        Bid/ask spread of the bar triggering the order. Defaults to 0.
    vol: float
        Volatility estimate for the current bar. Defaults to 0.
    volume: float
        Traded volume of the current bar. Defaults to 0.
    id: str
        Unique order identifier generated automatically when the order is
        instantiated.
    """

    symbol: str
    qty: float
    price: Optional[float] = None
    spread: float = 0.0
    vol: float = 0.0
    volume: float = 0.0
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
    """Applies slippage and fees in basis points.

    A custom ``slippage_fn`` can be provided to model more advanced
    behaviors. The function receives the current bar's ``spread``,
    ``vol`` (volatility) and ``volume`` and returns slippage in basis
    points.
    """

    def __init__(
        self,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        slippage_fn: Optional[Callable[[float, float, float], float]] = None,
    ):
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.slippage_fn = slippage_fn

    def apply(
        self,
        price: float,
        qty: float,
        spread: float = 0.0,
        vol: float = 0.0,
        volume: float = 0.0,
    ) -> float:
        """Return total cost including slippage and fees."""

        slip_bps = (
            self.slippage_fn(spread, vol, volume)
            if self.slippage_fn is not None
            else self.slippage_bps
        )
        slip_price = price * (1 + slip_bps / 10_000)
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
        cost = self.cost_model.apply(
            order.price or 0.0,
            order.qty,
            order.spread,
            order.vol,
            order.volume,
        )
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

