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
=======
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Callable
import uuid
import threading
import time


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
    """Simple live-trading client with basic resiliency features.

    The implementation is intentionally lightweight, logging its actions
    instead of performing real network requests.  Nevertheless it models
    behaviours typically found in production clients such as periodic
    heartbeats, cancel-on-disconnect and retry logic with exponential
    backoff.
    """

    def __init__(
        self,
        heartbeat_interval: int = 30,
        timeout: int = 10,
        max_retries: int = 3,
        kill_switch: bool = False,
    ):
        self._positions: Dict[str, float] = {}
        self._open_orders: Set[str] = set()
        self._kill_switch_enabled = kill_switch
        self._killed = False
        self._heartbeat_interval = heartbeat_interval
        self._timeout = timeout
        self._max_retries = max_retries
        self._last_ack = datetime.now(timezone.utc)
        self._stop = threading.Event()
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _with_retries(self, fn: Callable, *a, **k):
        delay = 0.5
        for attempt in range(1, self._max_retries + 1):
            try:
                return fn(*a, **k)
            except Exception:  # pragma: no cover - network failure stub
                if attempt == self._max_retries:
                    raise
                time.sleep(delay)
                delay *= 2

    def _record_ack(self) -> None:
        self._last_ack = datetime.now(timezone.utc)

    def _send_heartbeat(self) -> None:
        print("HEARTBEAT")

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_interval):
            if datetime.now(timezone.utc) - self._last_ack > timedelta(seconds=self._timeout):
                self._on_disconnect()
                continue
            try:
                self._with_retries(self._send_heartbeat)
                self._record_ack()
            except Exception:
                pass

    def _on_disconnect(self) -> None:
        for oid in list(self._open_orders):
            try:
                self.cancel(oid)
            except Exception:
                pass
        self._open_orders.clear()
        if self._kill_switch_enabled:
            self._killed = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Stop the heartbeat thread."""

        self._stop.set()
        self._hb_thread.join(timeout=1)

    def send(self, order: Order) -> str:  # pragma: no cover - stub
        if self._killed:
            raise RuntimeError("kill switch activated")

        def _do_send() -> None:
            print(f"LIVE ORDER -> {order.symbol} {order.qty}@{order.price}")

        self._with_retries(_do_send)
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + order.qty
        self._open_orders.add(order.id)
        self._record_ack()
        return order.id

    def cancel(self, order_id: str) -> None:  # pragma: no cover - stub
        if self._killed:
            raise RuntimeError("kill switch activated")

        def _do_cancel() -> None:
            print(f"CANCEL ORDER -> {order_id}")

        self._with_retries(_do_cancel)
        self._open_orders.discard(order_id)
        self._record_ack()

    def positions(self) -> Dict[str, float]:  # pragma: no cover - trivial
        return dict(self._positions)

    def clock(self) -> datetime:  # pragma: no cover - trivial
        return datetime.now(timezone.utc)

