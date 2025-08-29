import logging

from quant_pipeline.risk import RiskManager


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
    new_weights = rm.apply_correlation_throttle(weights, corr=0.9)
    assert new_weights["BTC"] == 0.6 * 0.5
    assert new_weights["ETH"] == 0.4 * 0.5
