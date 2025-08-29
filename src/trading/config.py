from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import yaml
from dotenv import load_dotenv


@dataclass
class StableAllocConfig:
    pct: float = 0.50
    min_usd: float = 50
    cap_daily_usd: float = 2000
    ticker: str = "SPY"
    broker: str = "ibkr"
    timezone: str = "America/New_York"
    order_style: str = "market_open"


def load_stable_cfg(path: str | Path = Path("conf/stable_alloc.yaml")) -> StableAllocConfig:
    """Load stable allocation config from YAML and environment variables.

    Environment variables prefixed with ``STABLE_`` override the YAML values.
    Missing fields fall back to dataclass defaults.
    """
    dotenv_path = Path(".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    cfg = StableAllocConfig()
    cfg_path = Path(path)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for key, value in data.items():
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, value)

    env_map = {
        "pct": os.getenv("STABLE_PCT"),
        "min_usd": os.getenv("STABLE_MIN_USD"),
        "cap_daily_usd": os.getenv("STABLE_CAP_DAILY_USD"),
        "ticker": os.getenv("STABLE_TICKER"),
        "broker": os.getenv("STABLE_BROKER"),
        "timezone": os.getenv("STABLE_TIMEZONE"),
        "order_style": os.getenv("STABLE_ORDER_STYLE"),
    }
    for key, val in env_map.items():
        if val is None:
            continue
        current = getattr(cfg, key)
        if isinstance(current, float):
            setattr(cfg, key, float(val))
        elif isinstance(current, int):
            setattr(cfg, key, int(val))
        else:
            setattr(cfg, key, val)

    return cfg
