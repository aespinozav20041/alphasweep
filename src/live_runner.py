"""Minimal live trading runner using the unified engine."""
from __future__ import annotations

import yaml

from engine import TradingEngine
from execution import CostModel, BrokerExecutionClient
from risk import RiskLimits, RiskManager
from bt_runner import DummyModel, build_features


def main(cfg_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(open(cfg_path))
    risk = RiskManager(RiskLimits(max_position=cfg["risk"]["max_position"]))
    cost = CostModel(**cfg["execution"])
    exec_client = BrokerExecutionClient(cost_model=cost)
    engine = TradingEngine(DummyModel(), build_features, risk, exec_client)

    data = [
        {"symbol": "XYZ", "price": 100.0, "ma": 100.0},
        {"symbol": "XYZ", "price": 101.0, "ma": 100.5},
        {"symbol": "XYZ", "price": 99.5, "ma": 100.5},
    ]
    for bar in data:
        engine.on_bar(bar)

    print("Final positions:", exec_client.positions())


if __name__ == "__main__":  # pragma: no cover - manual run
    main()
