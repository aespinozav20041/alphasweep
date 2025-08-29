"""Append-only order ledger backed by SQLite.

This module replaces the previous in-memory list with a lightweight SQLite
database so that order intents and outcomes are durably persisted.  Each order
is stored as a pair of events: an ``intent`` recorded *before* the broker call
and a subsequent result such as ``filled`` or ``error``.  Additional utility
functions are provided to replay unreconciled intents or inspect the full
ledger for auditing purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import sqlite3
import uuid
from pathlib import Path
from typing import Iterable, List, Optional

DB_PATH = Path("data/ledger.db")


@dataclass
class LedgerEntry:
    intent_id: str
    order_id: str | None
    ticker: str
    usd: float
    status: str
    date: date
    note: str | None = None
    scheduled_at: datetime | None = None


_conn: sqlite3.Connection | None = None


def _connection() -> sqlite3.Connection:
    """Return a SQLite connection, creating tables on first use."""

    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH)
        _conn.row_factory = sqlite3.Row
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_id TEXT NOT NULL,
                order_id TEXT,
                ticker TEXT NOT NULL,
                usd REAL NOT NULL,
                status TEXT NOT NULL,
                day TEXT NOT NULL,
                note TEXT,
                scheduled_at TEXT,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                usd REAL NOT NULL,
                day TEXT NOT NULL,
                note TEXT,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day TEXT NOT NULL,
                pnl REAL NOT NULL,
                model TEXT,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _conn.commit()
    return _conn


def _row_to_entry(row: sqlite3.Row) -> LedgerEntry:
    scheduled = datetime.fromisoformat(row["scheduled_at"]) if row["scheduled_at"] else None
    return LedgerEntry(
        intent_id=row["intent_id"],
        order_id=row["order_id"],
        ticker=row["ticker"],
        usd=float(row["usd"]),
        status=row["status"],
        date=date.fromisoformat(row["day"]),
        note=row["note"],
        scheduled_at=scheduled,
    )


def record(
    order_id: str,
    ticker: str,
    usd: float,
    status: str,
    day: Optional[date] = None,
    note: str | None = None,
    scheduled_at: datetime | None = None,
) -> None:
    """Record a generic ledger entry without an explicit intent.

    This is used for events such as sweep blocks where no broker order is
    attempted.  A fresh ``intent_id`` is generated automatically.
    """

    if day is None:
        day = datetime.utcnow().date()
    conn = _connection()
    conn.execute(
        """
        INSERT INTO ledger(intent_id, order_id, ticker, usd, status, day, note, scheduled_at)
        VALUES(?,?,?,?,?,?,?,?)
        """,
        (
            uuid.uuid4().hex,
            order_id,
            ticker,
            usd,
            status,
            day.isoformat(),
            note,
            scheduled_at.isoformat() if scheduled_at else None,
        ),
    )
    conn.commit()


def record_intent(
    ticker: str,
    usd: float,
    *,
    day: Optional[date] = None,
    note: str | None = None,
    scheduled_at: datetime | None = None,
    status: str = "intent",
) -> str:
    """Record an order intent and return its identifier."""

    if day is None:
        day = datetime.utcnow().date()
    intent_id = uuid.uuid4().hex
    conn = _connection()
    conn.execute(
        """
        INSERT INTO ledger(intent_id, order_id, ticker, usd, status, day, note, scheduled_at)
        VALUES(?,?,?,?,?,?,?,?)
        """,
        (
            intent_id,
            None,
            ticker,
            usd,
            status,
            day.isoformat(),
            note,
            scheduled_at.isoformat() if scheduled_at else None,
        ),
    )
    conn.commit()
    return intent_id


def record_result(
    intent_id: str,
    order_id: str,
    status: str,
    *,
    day: Optional[date] = None,
    note: str | None = None,
) -> None:
    """Append a result entry for a previously recorded intent."""

    if day is None:
        day = datetime.utcnow().date()
    conn = _connection()
    conn.execute(
        """
        INSERT INTO ledger(intent_id, order_id, ticker, usd, status, day, note, scheduled_at)
        SELECT ?, ?, ticker, usd, ?, ?, ?, scheduled_at FROM ledger
        WHERE intent_id = ? AND status IN ('intent', 'planned')
        LIMIT 1
        """,
        (
            intent_id,
            order_id,
            status,
            day.isoformat(),
            note,
            intent_id,
        ),
    )
    conn.commit()


def record_fill(
    order_id: str,
    ticker: str,
    qty: float,
    price: float,
    *,
    day: Optional[date] = None,
    note: str | None = None,
) -> None:
    """Record a trade fill in the fills table."""

    if day is None:
        day = datetime.utcnow().date()
    conn = _connection()
    usd = qty * price
    conn.execute(
        """
        INSERT INTO fills(order_id, ticker, qty, price, usd, day, note)
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            order_id,
            ticker,
            float(qty),
            float(price),
            float(usd),
            day.isoformat(),
            note,
        ),
    )
    conn.commit()


def record_pnl(
    pnl: float,
    *,
    day: Optional[date] = None,
    model: str | None = None,
) -> None:
    """Record a PnL observation for a given day and optional model."""

    if day is None:
        day = datetime.utcnow().date()
    conn = _connection()
    conn.execute(
        "INSERT INTO pnl(day, pnl, model) VALUES(?,?,?)",
        (day.isoformat(), float(pnl), model),
    )
    conn.commit()


def entries() -> List[LedgerEntry]:
    """Return all ledger entries in insertion order."""

    conn = _connection()
    rows = conn.execute("SELECT intent_id, order_id, ticker, usd, status, day, note, scheduled_at FROM ledger ORDER BY id").fetchall()
    return [_row_to_entry(r) for r in rows]


def pending(status: str = "planned") -> List[LedgerEntry]:
    """Return intents with the given status lacking a corresponding result."""

    conn = _connection()
    rows = conn.execute(
        """
        SELECT intent_id, order_id, ticker, usd, status, day, note, scheduled_at
        FROM ledger AS l
        WHERE status = ?
          AND intent_id NOT IN (
                SELECT intent_id FROM ledger WHERE status NOT IN ('intent', 'planned')
            )
        ORDER BY id
        """,
        (status,),
    ).fetchall()
    return [_row_to_entry(r) for r in rows]


def replay() -> Iterable[LedgerEntry]:
    """Yield ledger entries in chronological order for reconstruction."""

    for entry in entries():
        yield entry


def clear() -> None:
    """Remove all ledger entries (primarily for tests)."""

    conn = _connection()
    conn.execute("DELETE FROM ledger")
    conn.execute("DELETE FROM fills")
    conn.execute("DELETE FROM pnl")
    conn.commit()


__all__ = [
    "LedgerEntry",
    "record",
    "record_intent",
    "record_result",
    "record_fill",
    "record_pnl",
    "entries",
    "pending",
    "replay",
    "clear",
]

