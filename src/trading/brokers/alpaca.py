"""Alpaca Markets execution helpers using the official SDK."""

from __future__ import annotations

import uuid
from typing import Tuple, Any

from .. import ledger
from quant_pipeline.observability import Observability

MIN_NOTIONAL = 1.0


def _check(usd: float) -> None:
    if usd < MIN_NOTIONAL:
        raise ValueError("notional below minNotional")


def _client() -> Any:
    """Instantiate Alpaca REST client.

    Import is deferred so tests can monkeypatch ``_client`` without requiring
    the SDK to be installed at import time.
    """

    import alpaca_trade_api as tradeapi  # type: ignore

    return tradeapi.REST()


def _record(
    order_id: str,
    ticker: str,
    usd: float,
    status: str,
    obs: Observability | None,
) -> Tuple[str, str]:
    """Record order details to ledger and observability."""

    ledger.record(order_id, ticker, usd, status)
    if obs is not None:
        if status.lower() in {"canceled", "rejected", "error"}:
            obs.increment_order_errors()
        else:
            obs.increment_orders_sent()
    return order_id, status


def place_market_on_open(
    ticker: str, usd: float, obs: Observability | None = None
) -> Tuple[str, str]:
    """Submit a market-on-open order via Alpaca."""

    _check(usd)
    client = _client()
    try:
        order = client.submit_order(
            symbol=ticker,
            notional=usd,
            side="buy",
            type="market",
            time_in_force="opg",
        )
    except Exception as exc:  # pragma: no cover - network errors mocked in tests
        if obs is not None:
            obs.increment_order_errors()
        err_id = f"alpaca-error-{uuid.uuid4().hex[:8]}"
        ledger.record(err_id, ticker, usd, "error", note=str(exc))
        raise
    order_id = getattr(order, "id", f"alpaca-{uuid.uuid4().hex[:8]}")
    status = getattr(order, "status", "unknown")
    return _record(order_id, ticker, usd, status, obs)


def place_twap(
    ticker: str, usd: float, minutes: int = 30, obs: Observability | None = None
) -> Tuple[str, str]:
    """Submit a TWAP order via Alpaca."""

    _check(usd)
    client = _client()
    try:
        order = client.submit_order(
            symbol=ticker,
            notional=usd,
            side="buy",
            type="market",
            time_in_force="day",
            client_order_id=f"twap-{minutes}",
        )
    except Exception as exc:  # pragma: no cover - network errors mocked in tests
        if obs is not None:
            obs.increment_order_errors()
        err_id = f"alpaca-error-{uuid.uuid4().hex[:8]}"
        ledger.record(err_id, ticker, usd, "error", note=str(exc))
        raise
    order_id = getattr(order, "id", f"alpaca-{uuid.uuid4().hex[:8]}")
    status = getattr(order, "status", "unknown")
    return _record(order_id, ticker, usd, status, obs)


def place_limit_vwap(
    ticker: str, usd: float, obs: Observability | None = None
) -> Tuple[str, str]:
    """Submit a limit VWAP order via Alpaca."""

    _check(usd)
    client = _client()
    try:
        order = client.submit_order(
            symbol=ticker,
            notional=usd,
            side="buy",
            type="limit",
            time_in_force="day",
            client_order_id="vwap",
            limit_price=usd,  # placeholder
        )
    except Exception as exc:  # pragma: no cover - network errors mocked in tests
        if obs is not None:
            obs.increment_order_errors()
        err_id = f"alpaca-error-{uuid.uuid4().hex[:8]}"
        ledger.record(err_id, ticker, usd, "error", note=str(exc))
        raise
    order_id = getattr(order, "id", f"alpaca-{uuid.uuid4().hex[:8]}")
    status = getattr(order, "status", "unknown")
    return _record(order_id, ticker, usd, status, obs)
