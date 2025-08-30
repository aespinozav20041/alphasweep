"""Minimal walk-forward example using the unified engine.

A real walk-forward pipeline would retrain the model on each in-sample
window.  For simplicity this example just loops over data and resets the
engine to illustrate how :class:`SimExecutionClient` can be reused.
"""

from __future__ import annotations

import yaml

from engine import TradingEngine
from execution import CostModel, SimExecutionClient
from risk import RiskLimits, RiskManager
from bt_runner import DummyModel, build_features


def main(cfg_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(open(cfg_path))
    risk_cfg = cfg.get("risk", {})
    trail_cfg = risk_cfg.get("trailing", {})
    limits = RiskLimits(
        max_position=risk_cfg.get("max_position", 0.0),
        trailing_enabled=trail_cfg.get("enabled", False),
        atr_mult_trail=trail_cfg.get("atr_mult_trail", 3.0),
    )
    risk = RiskManager(limits)
    cost = CostModel(**cfg["execution"])
    exec_client = SimExecutionClient(cost_model=cost)
    engine = TradingEngine(DummyModel(), build_features, risk, exec_client)

    # Split data into two folds -------------------------------------------
    data = [
        {"symbol": "XYZ", "price": 100.0, "ma": 100.0},
        {"symbol": "XYZ", "price": 102.0, "ma": 101.0},  # first fold
        {"symbol": "XYZ", "price": 101.0, "ma": 101.5},
        {"symbol": "XYZ", "price": 103.0, "ma": 102.0},  # second fold
    ]
    split = len(data) // 2
    folds = [data[:split], data[split:]]

    for i, fold in enumerate(folds):
        print(f"Running fold {i}")
        for bar in fold:
            engine.on_bar(bar)

    print("Final positions:", exec_client.positions())


if __name__ == "__main__":  # pragma: no cover - manual run
    main()

