from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional


@dataclass
class LedgerEntry:
    order_id: str
    ticker: str
    usd: float
    status: str
    date: date
    note: str | None = None
    scheduled_at: datetime | None = None


entries: List[LedgerEntry] = []


def record(
    order_id: str,
    ticker: str,
    usd: float,
    status: str,
    day: Optional[date] = None,
    note: str | None = None,
    scheduled_at: datetime | None = None,
) -> None:
    """Record a ledger entry.

    Parameters
    ----------
    order_id: str
        Broker-provided identifier.
    ticker: str
        Instrument ticker.
    usd: float
        Notional in USD.
    status: str
        Fill status.
    day: date, optional
        Settlement date. Defaults to current UTC date.
    note: str, optional
        Additional commentary about the entry.
    scheduled_at: datetime, optional
        Planned execution time for unsent orders.
    """

    if day is None:
        day = datetime.utcnow().date()
    entries.append(LedgerEntry(order_id, ticker, usd, status, day, note, scheduled_at))


def clear() -> None:
    entries.clear()
