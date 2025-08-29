import os
from pathlib import Path

from trading.config import load_stable_cfg


def test_load_stable_cfg_defaults(monkeypatch, tmp_path):
    # create minimal YAML and empty .env to ensure defaults fill in
    yaml_path = tmp_path / "stable_alloc.yaml"
    yaml_path.write_text("pct: 0.25\n")
    (tmp_path / ".env").write_text("")
    monkeypatch.chdir(tmp_path)
    cfg = load_stable_cfg(yaml_path)
    assert cfg.pct == 0.25
    # remaining fields use defaults
    assert cfg.min_usd == 50
    assert cfg.cap_daily_usd == 2000
    assert cfg.ticker == "SPY"
    assert cfg.broker == "ibkr"
    assert cfg.timezone == "America/New_York"
    assert cfg.order_style == "market_open"


def test_load_stable_cfg_env_override(monkeypatch, tmp_path):
    yaml_path = tmp_path / "stable_alloc.yaml"
    yaml_path.write_text("pct: 0.10\nmin_usd: 10\n")
    env_file = tmp_path / ".env"
    env_file.write_text("STABLE_TICKER=VOO\nSTABLE_PCT=0.7\n")
    monkeypatch.chdir(tmp_path)
    cfg = load_stable_cfg(yaml_path)
    assert cfg.pct == 0.7  # env overrides yaml
    assert cfg.ticker == "VOO"
    assert cfg.min_usd == 10  # yaml overrides default
