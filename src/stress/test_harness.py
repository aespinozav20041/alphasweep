from __future__ import annotations
"""Simple stress testing harness.

This module provides a lightweight framework to run deterministic stress
scenarios against the trading engine.  Scenarios are defined by multiplicative
factors on latency, fees and spreads as well as optional trading halt windows
and random gaps in the return series.  The harness produces both CSV and SQLite
reports along with a small Markdown summary.

The actual trading engine is not exercised in the unit tests; instead we
simulate P&L series so that the harness remains fast and deterministic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import sqlite3

import numpy as np
import pandas as pd
import yaml


@dataclass
class StressScenario:
    """Definition of a stress test scenario."""

    name: str
    latency_factor: float = 1.0
    fees_factor: float = 1.0
    spread_factor: float = 1.0
    trading_halts: Sequence[tuple[int, int]] | None = None
    random_gaps: bool = False


def _simulate_returns(seed: int = 0, n: int = 100) -> np.ndarray:
    """Generate a deterministic baseline return series."""

    rng = np.random.default_rng(seed)
    return rng.normal(0.001, 0.01, size=n)


def _apply_scenario(r: np.ndarray, sc: StressScenario) -> np.ndarray:
    """Apply scenario transformations to returns."""

    series = r.copy()
    # trading halts zero out returns for the given index windows
    if sc.trading_halts:
        for start, end in sc.trading_halts:
            series[start:end] = 0.0
    # random gaps drop 10% of observations
    if sc.random_gaps:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(series), int(0.1 * len(series)), replace=False)
        series[idx] = 0.0
    # cost impact: each factor >1 introduces additional cost per trade
    cost = 0.0001 * ((sc.fees_factor - 1.0) + (sc.spread_factor - 1.0) + (sc.latency_factor - 1.0))
    series = series - cost
    return series


def _sharpe(r: np.ndarray) -> float:
    if r.std() == 0:
        return 0.0
    return r.mean() / r.std() * np.sqrt(252)


def run_stress_scenarios(engine_cfg: dict, scenarios: Iterable[StressScenario], out_dir: Path | str = "reports") -> pd.DataFrame:
    """Run the provided stress scenarios.

    Parameters
    ----------
    engine_cfg:
        Dictionary with engine configuration (unused but kept for API parity).
    scenarios:
        Iterable of :class:`StressScenario` definitions.
    out_dir:
        Destination directory for reports.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_returns = _simulate_returns()
    records = []
    for sc in scenarios:
        stressed = _apply_scenario(base_returns, sc)
        sharpe = _sharpe(stressed)
        records.append({"scenario": sc.name, "sharpe": sharpe})

    df = pd.DataFrame(records)
    csv_path = out_path / "stress_results.csv"
    df.to_csv(csv_path, index=False)

    db_path = out_path / "stress_results.db"
    with sqlite3.connect(db_path) as conn:
        df.to_sql("stress_runs", conn, if_exists="replace", index=False)

    try:
        table = df.to_markdown(index=False)
    except Exception:  # pragma: no cover - optional dependency
        table = df.to_string(index=False)
    md_lines = ["# Stress Test Summary", "", table]
    (out_path / "stress_summary.md").write_text("\n".join(md_lines))
    return df


def _load_config(path: Path) -> tuple[dict, list[StressScenario]]:
    """Parse YAML configuration file."""

    cfg = yaml.safe_load(Path(path).read_text())
    engine_cfg = cfg.get("engine", {})
    scenarios_cfg = cfg.get("scenarios", [])
    scenarios = []
    for sc in scenarios_cfg:
        scenarios.append(
            StressScenario(
                name=sc["name"],
                latency_factor=sc.get("latency_factor", 1.0),
                fees_factor=sc.get("fees_factor", 1.0),
                spread_factor=sc.get("spread_factor", 1.0),
                trading_halts=[tuple(x) for x in sc.get("trading_halts", [])] or None,
                random_gaps=sc.get("random_gaps", False),
            )
        )
    return engine_cfg, scenarios


def main(args: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI
    import argparse

    p = argparse.ArgumentParser(description="Run stress test scenarios")
    p.add_argument("--config", required=True, help="YAML config with scenarios")
    ns = p.parse_args(args)
    engine_cfg, scenarios = _load_config(Path(ns.config))
    run_stress_scenarios(engine_cfg, scenarios)


if __name__ == "__main__":  # pragma: no cover
    main()
