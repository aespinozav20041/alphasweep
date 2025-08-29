import logging

import logging
from pathlib import Path

import pytest

from quant_pipeline.risk import RiskManager, load_risk_config


def _rm(position_closer=None):
    return RiskManager(
        max_dd_daily=0.05,
        max_dd_weekly=0.1,
        latency_threshold=100,
        latency_window=3,
        pause_minutes=1,
        max_position_notional_symbol={"BTC": 1000},
        max_total_notional=5000,
        max_turnover_day=2000,
        min_notional_exchange={"binance": 10},
        position_closer=position_closer,
    )


def test_drawdown_kill_switch(caplog):
    closed = False

    def close():
        nonlocal closed
        closed = True

    rm = _rm(close)
    with caplog.at_level(logging.WARNING):
        rm.update_drawdowns(0.06, 0.0)
    assert closed
    assert rm.is_paused()
    assert "drawdown threshold exceeded" in caplog.text
    assert not rm.validate_order(
        symbol="BTC", notional=10, total_notional=0, turnover_day=0, exchange="binance"
    )


def test_latency_kill_switch(caplog):
    closed = False

    def close():
        nonlocal closed
        closed = True

    rm = _rm(close)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            rm.record_latency(150)
    assert closed
    assert rm.is_paused()
    assert "latency threshold exceeded" in caplog.text


def test_limit_enforcement():
    rm = RiskManager(
        max_dd_daily=0.2,
        max_dd_weekly=0.2,
        latency_threshold=100,
        latency_window=2,
        pause_minutes=1,
        max_position_notional_symbol={"BTC": 100},
        max_total_notional=200,
        max_turnover_day=150,
        min_notional_exchange={"binance": 50},
    )
    assert not rm.validate_order(
        symbol="BTC", notional=200, total_notional=0, turnover_day=0, exchange="binance"
    )
    assert not rm.validate_order(
        symbol="ETH", notional=10, total_notional=0, turnover_day=0, exchange="binance"
    )
    assert not rm.validate_order(
        symbol="BTC", notional=50, total_notional=160, turnover_day=0, exchange="binance"
    )
    assert not rm.validate_order(
        symbol="BTC", notional=50, total_notional=0, turnover_day=120, exchange="binance"
    )
    assert rm.validate_order(
        symbol="BTC", notional=50, total_notional=0, turnover_day=0, exchange="binance"
    )


def test_correlation_throttle():
    rm = RiskManager(
        max_dd_daily=1,
        max_dd_weekly=1,
        latency_threshold=100,
        latency_window=1,
        pause_minutes=1,
    )
    weights = {"BTC": 0.6, "ETH": 0.4}
    new_weights = rm.apply_correlation_throttle(weights, corr=0.9, regime="bull")
    assert new_weights["BTC"] == 0.6 * 0.5
    assert new_weights["ETH"] == 0.4 * 0.5


def test_kelly_atr_and_regime_throttle():
    rm = RiskManager(
        max_dd_daily=1,
        max_dd_weekly=1,
        latency_threshold=100,
        latency_window=1,
        pause_minutes=1,
        target_volatility=0.02,
        kelly_fraction=0.5,
        sl_atr=3,
        tp_atr=6,
        regime_reduction=0.3,
    )
    pos = rm.kelly_position(mu=0.02, sigma=0.1)
    assert pos == pytest.approx(0.2)
    sl, tp = rm.atr_sl_tp(100, 2)
    assert sl == pytest.approx(94)
    assert tp == pytest.approx(112)
    weights = {"BTC": 1.0}
    new_weights = rm.apply_correlation_throttle(weights, corr=0.0, regime="bear")
    assert new_weights["BTC"] == pytest.approx(0.3)


def test_load_risk_config(tmp_path):
    path = tmp_path / "risk.yaml"
    path.write_text("max_dd_daily: 0.07\npause_minutes: 5\n")
    cfg = load_risk_config(path)
    assert cfg.max_dd_daily == 0.07
    assert cfg.pause_minutes == 5
