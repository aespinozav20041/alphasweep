from __future__ import annotations
"""Champion–Challenger framework.

Keeps track of metrics for multiple models and promotes challengers to
champion status when they meet predefined thresholds.  When a promotion occurs
it writes a Markdown report and logs the event in SQLite.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable
import sqlite3
import yaml


@dataclass
class ModelMetrics:
    """Container with evaluation metrics for a model."""

    sharpe_is: float
    sharpe_oos: float
    calmar_oos: float
    tracking_error_live: float


class ChampionChallenger:
    """Manage champion/challenger promotion."""

    def __init__(self, sharpe_oos: float, calmar_oos: float, tracking_error_live: float, out_dir: Path | str = "reports") -> None:
        self.thresholds = {
            "sharpe_oos": sharpe_oos,
            "calmar_oos": calmar_oos,
            "tracking_error_live": tracking_error_live,
        }
        self.models: Dict[str, ModelMetrics] = {}
        self.champion: str | None = None
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def register(self, name: str, metrics: ModelMetrics) -> None:
        self.models[name] = metrics

    def evaluate(self) -> str | None:
        """Evaluate challengers and promote if criteria are met."""
        promoted: str | None = None
        for name, m in self.models.items():
            if (
                m.sharpe_oos >= self.thresholds["sharpe_oos"]
                and m.calmar_oos >= self.thresholds["calmar_oos"]
                and m.tracking_error_live <= self.thresholds["tracking_error_live"]
            ):
                promoted = name
                break
        if promoted and promoted != self.champion:
            previous = self.champion
            self.champion = promoted
            self._write_report(previous, promoted, self.models[promoted])
            self._record_promotion(previous, promoted)
        return self.champion

    def _write_report(self, old: str | None, new: str, metrics: ModelMetrics) -> None:
        ts = datetime.utcnow().isoformat()
        lines = ["# Promotion Report", "", f"Timestamp: {ts}", "", f"New champion: **{new}**"]
        if old:
            lines.append(f"Previous champion: {old}")
        lines.extend([
            "", "## Metrics", "",
            f"- Sharpe OOS: {metrics.sharpe_oos}",
            f"- Calmar OOS: {metrics.calmar_oos}",
            f"- TrackingError live: {metrics.tracking_error_live}",
        ])
        (self.out_dir / "promotion_report.md").write_text("\n".join(lines))

    def _record_promotion(self, old: str | None, new: str) -> None:
        db = self.out_dir / "promotion_events.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS promotion_events(ts TEXT, old TEXT, new TEXT)")
            conn.execute("INSERT INTO promotion_events(ts, old, new) VALUES (?,?,?)", (datetime.utcnow().isoformat(), old, new))
            conn.commit()


def _load_config(path: Path) -> tuple[ChampionChallenger, Dict[str, ModelMetrics]]:
    cfg = yaml.safe_load(Path(path).read_text())
    thr = cfg["thresholds"]
    cc = ChampionChallenger(
        sharpe_oos=thr["sharpe_oos"],
        calmar_oos=thr["calmar_oos"],
        tracking_error_live=thr["tracking_error_live"],
    )
    models = {}
    for m in cfg.get("models", []):
        models[m["name"]] = ModelMetrics(
            sharpe_is=m.get("sharpe_is", 0.0),
            sharpe_oos=m.get("sharpe_oos", 0.0),
            calmar_oos=m.get("calmar_oos", 0.0),
            tracking_error_live=m.get("tracking_error_live", 0.0),
        )
    return cc, models


def main(args: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI
    import argparse
    p = argparse.ArgumentParser(description="Run Champion–Challenger evaluation")
    p.add_argument("--config", required=True, help="Path to YAML config")
    ns = p.parse_args(args)
    cc, models = _load_config(Path(ns.config))
    for name, metrics in models.items():
        cc.register(name, metrics)
    cc.evaluate()


if __name__ == "__main__":  # pragma: no cover
    main()
