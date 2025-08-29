"""Live trading runner using :class:`TradingEngine`.

This script wires the engine with a :class:`BrokerExecutionClient` and a
placeholder live data subscription.  The configuration is read from
``config.yaml`` and must set ``mode: live``.
"""

from __future__ import annotations

import random
import time
from typing import Iterator

import yaml

from engine import TradingEngine
from execution import BrokerExecutionClient
from risk import RiskLimits, RiskManager
from bt_runner import DummyModel, build_features


# ---------------------------------------------------------------------------
# Data feed
# ---------------------------------------------------------------------------

def subscribe_live_data(symbol: str) -> Iterator[dict]:
    """Yield live market data bars for ``symbol``.

    In production this function would connect to a broker or exchange
    streaming API (e.g. websockets).  The current implementation simply
    generates random walk prices to illustrate the integration with the
    trading engine.
    """

    price = 100.0
    ma = price
    while True:
        # Random walk for demonstration purposes
        price += random.uniform(-1, 1)
        ma = ma * 0.9 + price * 0.1
        yield {"symbol": symbol, "price": price, "ma": ma}
        time.sleep(1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main(cfg_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(open(cfg_path))
    if cfg.get("mode") != "live":
        raise ValueError("config.yaml must set mode=live for live trading")

    risk = RiskManager(RiskLimits(max_position=cfg["risk"]["max_position"]))
    exec_client = BrokerExecutionClient()
    engine = TradingEngine(DummyModel(), build_features, risk, exec_client)

    for bar in subscribe_live_data("XYZ"):
        engine.on_bar(bar)


if __name__ == "__main__":  # pragma: no cover - manual run
    try:
        main()
    except KeyboardInterrupt:
        pass
