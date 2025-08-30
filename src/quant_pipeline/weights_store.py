from __future__ import annotations

"""Persistence of strategy weights in SQLite."""

import sqlite3
from pathlib import Path
from typing import Dict, Mapping


def save_weights(db_path: str | Path, weights: Mapping[str, Mapping[str, Mapping[str, float]]]) -> None:
    """Persist nested weight mapping to a SQLite database."""

    path = Path(db_path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS weights(
            strategy TEXT,
            symbol TEXT,
            horizon TEXT,
            weight REAL,
            PRIMARY KEY(strategy, symbol, horizon)
        )
        """
    )
    for strat, symbols in weights.items():
        for symbol, horizons in symbols.items():
            for horizon, w in horizons.items():
                cur.execute(
                    "REPLACE INTO weights(strategy, symbol, horizon, weight) VALUES (?, ?, ?, ?)",
                    (strat, symbol, horizon, float(w)),
                )
    conn.commit()
    conn.close()


def load_weights(db_path: str | Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load weights from ``db_path``."""

    path = Path(db_path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS weights(
            strategy TEXT,
            symbol TEXT,
            horizon TEXT,
            weight REAL,
            PRIMARY KEY(strategy, symbol, horizon)
        )
        """
    )
    rows = cur.execute("SELECT strategy, symbol, horizon, weight FROM weights").fetchall()
    conn.close()
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for strat, symbol, horizon, w in rows:
        out.setdefault(strat, {}).setdefault(symbol, {})[horizon] = float(w)
    return out


__all__ = ["save_weights", "load_weights"]
