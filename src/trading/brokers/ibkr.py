"""Interactive Brokers execution helpers using the official SDK."""

from __future__ import annotations

import uuid
from typing import Tuple, Any

from .. import ledger
from quant_pipeline.observability import Observability

MIN_NOTIONAL = 50.0


def _check(usd: float) -> None:
    if usd < MIN_NOTIONAL:
        raise ValueError("notional below minNotional")


def _client() -> Any:
    """Instantiate ``ib_insync.IB`` connection."""

    from ib_insync import IB  # type: ignore

    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=1)
    return ib


def _record_result(
    intent_id: str,
    ticker: str,
    usd: float,
    order_id: str,
    status: str,
    obs: Observability | None,
    *,
    day=None,
    note: str | None = None,
) -> Tuple[str, str]:
    ledger.record_result(intent_id, order_id, status, day=day, note=note)
    if obs is not None:
        if status.lower() in {"cancelled", "rejected", "error"}:
            obs.increment_order_errors()
        else:
            obs.increment_orders_sent()
    return order_id, status


def place_market_on_open(
    ticker: str,
    usd: float,
    obs: Observability | None = None,
    *,
    day=None,
    intent_id: str | None = None,
) -> Tuple[str, str]:
    """Submit a market-on-open order via IBKR."""

    _check(usd)
    if intent_id is None:
        intent_id = ledger.record_intent(ticker, usd, day=day)
    ib = _client()
    try:
        from ib_insync import Stock, MarketOrder  # type: ignore

        contract = Stock(ticker, "SMART", "USD")
        order = MarketOrder("BUY", usd, tif="OPG")
        trade = ib.placeOrder(contract, order)
    except Exception as exc:  # pragma: no cover - network errors mocked in tests
        if obs is not None:
            obs.increment_order_errors()
        err_id = f"ibkr-error-{uuid.uuid4().hex[:8]}"
        ledger.record_result(intent_id, err_id, "error", day=day, note=str(exc))
        raise
    order_id = str(getattr(trade.order, "orderId", uuid.uuid4().hex))
    status = str(getattr(trade.orderStatus, "status", "unknown"))
    return _record_result(intent_id, ticker, usd, order_id, status, obs, day=day)


def place_twap(
    ticker: str,
    usd: float,
    minutes: int = 30,
    obs: Observability | None = None,
    *,
    day=None,
    intent_id: str | None = None,
) -> Tuple[str, str]:
    """Submit a TWAP order via IBKR."""

    _check(usd)
    if intent_id is None:
        intent_id = ledger.record_intent(ticker, usd, day=day)
    ib = _client()
    try:
        from ib_insync import Stock, MarketOrder, TagValue  # type: ignore

        contract = Stock(ticker, "SMART", "USD")
        order = MarketOrder("BUY", usd)
        order.algoStrategy = "TWAP"
        order.algoParams = [TagValue("time", str(minutes))]
        trade = ib.placeOrder(contract, order)
    except Exception as exc:  # pragma: no cover - network errors mocked in tests
        if obs is not None:
            obs.increment_order_errors()
        err_id = f"ibkr-error-{uuid.uuid4().hex[:8]}"
        ledger.record_result(intent_id, err_id, "error", day=day, note=str(exc))
        raise
    order_id = str(getattr(trade.order, "orderId", uuid.uuid4().hex))
    status = str(getattr(trade.orderStatus, "status", "unknown"))
    return _record_result(intent_id, ticker, usd, order_id, status, obs, day=day)


def place_limit_vwap(
    ticker: str,
    usd: float,
    obs: Observability | None = None,
    *,
    day=None,
    intent_id: str | None = None,
) -> Tuple[str, str]:
    """Submit a limit VWAP order via IBKR."""

    _check(usd)
    if intent_id is None:
        intent_id = ledger.record_intent(ticker, usd, day=day)
    ib = _client()
    try:
        from ib_insync import Stock, LimitOrder  # type: ignore

        contract = Stock(ticker, "SMART", "USD")
        order = LimitOrder("BUY", usd, usd)
        order.algoStrategy = "VWAP"
        trade = ib.placeOrder(contract, order)
    except Exception as exc:  # pragma: no cover - network errors mocked in tests
        if obs is not None:
            obs.increment_order_errors()
        err_id = f"ibkr-error-{uuid.uuid4().hex[:8]}"
        ledger.record_result(intent_id, err_id, "error", day=day, note=str(exc))
        raise
    order_id = str(getattr(trade.order, "orderId", uuid.uuid4().hex))
    status = str(getattr(trade.orderStatus, "status", "unknown"))
    return _record_result(intent_id, ticker, usd, order_id, status, obs, day=day)

