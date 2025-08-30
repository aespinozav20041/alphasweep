"""P&L attribution utilities.

This module provides helper functions to attribute daily and weekly P&L
by (model, horizon, symbol) as well as by factor exposures (volatility,
carry and momentum).  Results can be exported to CSV files and persisted
in a SQLite database for further analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable, Tuple

import pandas as pd


@dataclass
class AttributionResult:
    """Container for attribution results."""

    daily: pd.DataFrame
    weekly: pd.DataFrame


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def attribute_by_model(df: pd.DataFrame) -> AttributionResult:
    """Attribute P&L by model, horizon and symbol.

    Parameters
    ----------
    df:
        DataFrame containing at least the following columns:
        ``date``, ``model``, ``horizon``, ``symbol`` and ``pnl``.
    """
    grouped = df.groupby(["date", "model", "horizon", "symbol"], as_index=False)["pnl"].sum()
    # Weekly aggregation uses the ISO week (Monday start)
    df = df.copy()
    df["week"] = df["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
    weekly = df.groupby(["week", "model", "horizon", "symbol"], as_index=False)["pnl"].sum()
    weekly.rename(columns={"week": "date"}, inplace=True)
    return AttributionResult(daily=grouped.sort_values("date"), weekly=weekly.sort_values("date"))


def attribute_by_factor(df: pd.DataFrame) -> AttributionResult:
    """Attribute P&L by factor exposures (vol, carry, momentum)."""
    factors = ["vol", "carry", "momo"]
    daily = df.groupby("date")[factors].sum().reset_index().sort_values("date")
    df = df.copy()
    df["week"] = df["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
    weekly = df.groupby("week")[factors].sum().reset_index().sort_values("week")
    weekly.rename(columns={"week": "date"}, inplace=True)
    return AttributionResult(daily=daily, weekly=weekly)


def to_csv(res: AttributionResult, daily_path: Path, weekly_path: Path) -> None:
    _ensure_dir(daily_path)
    res.daily.to_csv(daily_path, index=False)
    res.weekly.to_csv(weekly_path, index=False)


def to_sqlite(res: AttributionResult, db_path: Path, daily_table: str, weekly_table: str) -> None:
    _ensure_dir(db_path)
    with sqlite3.connect(db_path) as conn:
        res.daily.to_sql(daily_table, conn, if_exists="replace", index=False)
        res.weekly.to_sql(weekly_table, conn, if_exists="replace", index=False)


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def run(trades_csv: Path, out_dir: Path = Path("reports"), db_path: Path | None = None) -> None:
    """Run P&L attribution for ``trades_csv``.

    Parameters
    ----------
    trades_csv:
        CSV containing trade level data with required columns.
    out_dir:
        Directory where CSV reports will be written.
    db_path:
        Optional path for SQLite DB. Defaults to ``out_dir / 'pnl_attribution.db'``.
    """
    out_dir = Path(out_dir)
    db_path = Path(db_path) if db_path else out_dir / "pnl_attribution.db"

    df = _load_csv(Path(trades_csv))
    model_attr = attribute_by_model(df)
    factor_attr = attribute_by_factor(df)

    # CSV exports
    to_csv(model_attr, out_dir / "pnl_attribution_daily.csv", out_dir / "pnl_attribution_weekly.csv")
    to_csv(factor_attr, out_dir / "factor_attribution_daily.csv", out_dir / "factor_attribution_weekly.csv")

    # SQLite exports
    to_sqlite(model_attr, db_path, "daily_model_attribution", "weekly_model_attribution")
    to_sqlite(factor_attr, db_path, "daily_factor_attribution", "weekly_factor_attribution")


def main(args: Iterable[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run P&L attribution")
    parser.add_argument("trades_csv", help="Input CSV with trade level data")
    parser.add_argument("--out", default="reports", help="Directory for CSV outputs")
    parser.add_argument("--db", default=None, help="Optional SQLite DB path")
    ns = parser.parse_args(args)
    run(Path(ns.trades_csv), Path(ns.out), Path(ns.db) if ns.db else None)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
