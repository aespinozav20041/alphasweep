"""Alpaca Markets execution helpers (stubbed)."""

from __future__ import annotations

import uuid
from typing import Tuple

MIN_NOTIONAL = 1.0


def _check(usd: float) -> None:
    if usd < MIN_NOTIONAL:
        raise ValueError("notional below minNotional")


def place_market_on_open(ticker: str, usd: float) -> Tuple[str, str]:
    _check(usd)
    order_id = f"alpaca-moo-{uuid.uuid4().hex[:8]}"
    return order_id, "submitted"


def place_twap(ticker: str, usd: float, minutes: int = 30) -> Tuple[str, str]:
    _check(usd)
    order_id = f"alpaca-twap-{uuid.uuid4().hex[:8]}"
    return order_id, "working"


def place_limit_vwap(ticker: str, usd: float) -> Tuple[str, str]:
    _check(usd)
    order_id = f"alpaca-lvwap-{uuid.uuid4().hex[:8]}"
    return order_id, "working"
