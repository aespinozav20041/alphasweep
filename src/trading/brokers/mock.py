"""Mock broker for development and tests."""

from __future__ import annotations

import uuid
from typing import Tuple

from .. import ledger

MIN_NOTIONAL = 1.0


def _check(usd: float) -> None:
    if usd < MIN_NOTIONAL:
        raise ValueError("notional below minNotional")


def _intent(ticker: str, usd: float, day, intent_id: str | None) -> str:
    return intent_id or ledger.record_intent(ticker, usd, day=day)


def place_market_on_open(
    ticker: str,
    usd: float,
    obs=None,
    *,
    day=None,
    intent_id: str | None = None,
) -> Tuple[str, str]:
    _check(usd)
    intent_id = _intent(ticker, usd, day, intent_id)
    order_id = f"mock-{uuid.uuid4().hex[:8]}"
    ledger.record_result(intent_id, order_id, "filled", day=day)
    return order_id, "filled"


def place_twap(
    ticker: str,
    usd: float,
    minutes: int = 30,
    obs=None,
    *,
    day=None,
    intent_id: str | None = None,
) -> Tuple[str, str]:
    _check(usd)
    intent_id = _intent(ticker, usd, day, intent_id)
    order_id = f"mock-twap-{uuid.uuid4().hex[:8]}"
    ledger.record_result(intent_id, order_id, "filled", day=day)
    return order_id, "filled"


def place_limit_vwap(
    ticker: str,
    usd: float,
    obs=None,
    *,
    day=None,
    intent_id: str | None = None,
) -> Tuple[str, str]:
    _check(usd)
    intent_id = _intent(ticker, usd, day, intent_id)
    order_id = f"mock-lvwap-{uuid.uuid4().hex[:8]}"
    ledger.record_result(intent_id, order_id, "filled", day=day)
    return order_id, "filled"
