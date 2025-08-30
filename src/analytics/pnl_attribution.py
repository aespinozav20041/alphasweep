from __future__ import annotations
"""P&L attribution utilities.

This module aggregates trade level data to attribute P&L by
(model/strategy/horizon/symbol) and by factor components such as volatility,
carry and momentum.  Results are written to CSV files, SQLite tables and simple
PNG charts for quick inspection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple
import sqlite3

import pandas as pd
import numpy as np


@dataclass
class AttributionTables:
    by_component: pd.DataFrame
    by_factor: pd.DataFrame


def compute_attribution(df: pd.DataFrame) -> AttributionTables:
    """Compute attribution tables from a trades DataFrame.

    Expected columns are ``date``, ``model``, ``strategy``, ``horizon``,
    ``symbol``, ``pnl``, ``vol``, ``carry`` and ``momo``.  The P&L column must
    equal ``vol + carry + momo`` for each row to make factor attribution
    consistent.
    """

    comp = df.groupby(["date", "model", "strategy", "horizon", "symbol"], as_index=False)["pnl"].sum()
    fac = df.groupby("date")[["vol", "carry", "momo"]].sum().reset_index()
    return AttributionTables(by_component=comp, by_factor=fac)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(res: AttributionTables, out: Path) -> None:
    _ensure_dir(out)
    res.by_component.to_csv(out / "pnl_attribution_components.csv", index=False)
    res.by_factor.to_csv(out / "pnl_attribution_factors.csv", index=False)


def _write_sqlite(res: AttributionTables, db: Path) -> None:
    _ensure_dir(db)
    with sqlite3.connect(db) as conn:
        res.by_component.to_sql("pnl_attrib", conn, if_exists="replace", index=False)
        res.by_factor.to_sql("pnl_attrib_factors", conn, if_exists="replace", index=False)


def _write_plots(res: AttributionTables, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return
    _ensure_dir(out / "pnl_factors.png")
    fig, ax = plt.subplots(figsize=(6, 3))
    res.by_factor.set_index("date").plot(kind="bar", ax=ax)
    ax.set_title("Factor Attribution")
    fig.tight_layout()
    fig.savefig(out / "pnl_factors.png")
    plt.close(fig)


def run(start: str, end: str, out_dir: Path | str = "reports", db_path: Path | None = None) -> AttributionTables:
    """Generate synthetic trades and perform attribution."""

    rng = np.random.default_rng(0)
    dates = pd.date_range(start, end, freq="D")
    models = ["m1", "m2"]
    strategies = ["edge_a", "edge_b"]
    horizons = ["h1", "h2"]
    symbols = ["AAA", "BBB"]
    rows = []
    for d in dates:
        for m in models:
            for st in strategies:
                for h in horizons:
                    for sym in symbols:
                        vol = rng.normal(0, 0.5)
                        carry = rng.normal(0, 0.2)
                        momo = rng.normal(0, 0.3)
                        pnl = vol + carry + momo
                        rows.append({
                            "date": d,
                            "model": m,
                            "strategy": st,
                            "horizon": h,
                            "symbol": sym,
                            "pnl": pnl,
                            "vol": vol,
                            "carry": carry,
                            "momo": momo,
                        })
    df = pd.DataFrame(rows)
    res = compute_attribution(df)

    out_path = Path(out_dir)
    _write_csv(res, out_path)
    _write_plots(res, out_path)
    db_path = db_path or out_path / "pnl_attrib.db"
    _write_sqlite(res, db_path)
    return res


def main(args: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI
    import argparse

    p = argparse.ArgumentParser(description="Run P&L attribution")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", default="reports")
    p.add_argument("--db", default=None)
    ns = p.parse_args(args)
    run(ns.start, ns.end, ns.out, Path(ns.db) if ns.db else None)


if __name__ == "__main__":  # pragma: no cover
    main()
