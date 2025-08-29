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
from typing import Dict, List, Optional, Sequence, Tuple
import uuid

import numpy as np


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
    def send(self, order: Order, symbol_state: Optional[Dict[str, float]] = None) -> str:
        """Send an order to the venue.

        Parameters
        ----------
        order:
            Order instance to be sent.
        symbol_state:
            Optional dictionary containing market state information such as
            spread, volatility and volume.

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
    """Applies linear slippage and fees in basis points.

    The model also supports a simple spread/volatility/volume based
    slippage estimate calibrated per symbol.
    """

    def __init__(
        self,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        coeffs: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ):
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.coeffs: Dict[str, Tuple[float, float, float]] = coeffs or {}

    # ------------------------------------------------------------------
    def spread_vol_volume(self, symbol_state: Dict[str, float], order: Order) -> float:
        """Return slippage estimate based on market state.

        Parameters
        ----------
        symbol_state:
            Dictionary with keys ``spread``, ``volatility`` and ``volume``.
        order:
            :class:`Order` instance for which the cost is being evaluated.
        """

        a, b, c = self.coeffs.get(order.symbol, (0.0, 0.0, 0.0))
        spread = symbol_state.get("spread", 0.0)
        vol = symbol_state.get("volatility", 0.0)
        volume = symbol_state.get("volume", float("inf"))
        if volume == 0:
            volume = float("inf")
        return a * spread + b * vol + c * abs(order.qty) / volume

    def calibrate(
        self,
        symbol: str,
        spreads: Sequence[float],
        volatilities: Sequence[float],
        qtys: Sequence[float],
        volumes: Sequence[float],
        slips: Sequence[float],
    ) -> Tuple[float, float, float]:
        """Calibrate ``a``, ``b`` and ``c`` for a symbol using least squares."""

        X = np.column_stack([spreads, volatilities, np.abs(qtys) / volumes])
        y = np.asarray(slips)
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = coeffs.tolist()
        self.coeffs[symbol] = (a, b, c)
        return self.coeffs[symbol]

    def apply(
        self,
        price: float,
        qty: float,
        symbol_state: Optional[Dict[str, float]] = None,
        order: Optional[Order] = None,
    ) -> float:
        """Return total cost including slippage and fees."""

        if symbol_state is not None and order is not None:
            slip = self.spread_vol_volume(symbol_state, order)
            slip_price = price + slip
        else:
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

    def send(
        self, order: Order, symbol_state: Optional[Dict[str, float]] = None
    ) -> str:  # pragma: no cover - trivial
        symbol_state = symbol_state or {}
        cost = self.cost_model.apply(order.price or 0.0, order.qty, symbol_state, order)
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

    def send(
        self, order: Order, symbol_state: Optional[Dict[str, float]] = None
    ) -> str:  # pragma: no cover - stub
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

