"""Example backtest runner using :class:`TradingEngine`.

The script wires together a dummy model and feature builder with the
:class:`SimExecutionClient` so that the same engine logic can later be
used for walk-forward or live trading.
"""

from __future__ import annotations

import yaml

from engine import TradingEngine
from execution import CostModel, SimExecutionClient
from risk import RiskLimits, RiskManager


class DummyModel:
    """Model with a ``predict`` method returning the provided feature."""

    def predict(self, features):  # pragma: no cover - trivial
        return features["signal"]


def build_features(bar):  # pragma: no cover - trivial
    # For illustration the feature is simply price momentum
    return {"signal": bar["price"] - bar.get("ma", bar["price"])}


def main(cfg_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(open(cfg_path))
    risk = RiskManager(RiskLimits(max_position=cfg["risk"]["max_position"]))
    cost = CostModel(**cfg["execution"])
    exec_client = SimExecutionClient(cost_model=cost)
    engine = TradingEngine(DummyModel(), build_features, risk, exec_client)

    # Example price series ------------------------------------------------
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

