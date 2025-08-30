"""SQLite-backed persistence for model signals and weights."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Tuple


class SignalStore:
    """Persist signals and weights by strategy/symbol/horizon."""

    def __init__(self, db_path: str | Path) -> None:
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS signals(
                ts INTEGER,
                strategy TEXT,
                symbol TEXT,
                horizon TEXT,
                signal REAL,
                weight REAL
            )
            """
        )
        self.conn.commit()

    def save(
        self,
        *,
        ts: int,
        strategy: str,
        symbol: str,
        horizon: str,
        signal: float,
        weight: float,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO signals(ts, strategy, symbol, horizon, signal, weight) VALUES(?,?,?,?,?,?)",
            (ts, strategy, symbol, horizon, signal, weight),
        )
        self.conn.commit()

    def load(
        self,
        *,
        strategy: str,
        symbol: str,
        horizon: str,
        start: int,
        end: int,
    ) -> list[Tuple[int, float, float]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT ts, signal, weight FROM signals
            WHERE strategy=? AND symbol=? AND horizon=? AND ts BETWEEN ? AND ?
            ORDER BY ts
            """,
            (strategy, symbol, horizon, start, end),
        )
        return [(int(ts), float(sig), float(w)) for ts, sig, w in cur.fetchall()]


__all__ = ["SignalStore"]
