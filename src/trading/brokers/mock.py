"""Mock broker for development and tests."""

from __future__ import annotations

import uuid
from typing import Tuple

from .. import ledger

MIN_NOTIONAL = 1.0


def _check(usd: float) -> None:
    if usd < MIN_NOTIONAL:
        raise ValueError("notional below minNotional")


def place_market_on_open(ticker: str, usd: float) -> Tuple[str, str]:
    _check(usd)
    order_id = f"mock-{uuid.uuid4().hex[:8]}"
    ledger.record(order_id, ticker, usd, "filled")
    return order_id, "filled"


def place_twap(ticker: str, usd: float, minutes: int = 30) -> Tuple[str, str]:
    _check(usd)
    order_id = f"mock-twap-{uuid.uuid4().hex[:8]}"
    ledger.record(order_id, ticker, usd, "filled")
    return order_id, "filled"


def place_limit_vwap(ticker: str, usd: float) -> Tuple[str, str]:
    _check(usd)
    order_id = f"mock-lvwap-{uuid.uuid4().hex[:8]}"
    ledger.record(order_id, ticker, usd, "filled")
    return order_id, "filled"
