import pandas as pd

from quant_pipeline.backtest import run_backtest


def test_run_backtest_returns_metrics():
    df = pd.DataFrame(
        {
            "ret": [0.01, -0.02, 0.03, -0.01],
            "spread": [0.001] * 4,
            "volume": [100] * 4,
        }
    )
    metrics = run_backtest(df, return_metrics=True)
    assert {"pnl", "calmar", "max_drawdown", "turnover"} <= metrics.keys()


def test_costs_reduce_pnl():
    df = pd.DataFrame({"ret": [0.01, 0.02, -0.01, 0.005]})
    base = run_backtest(df, return_metrics=True)["pnl"]
    costly = run_backtest(df, return_metrics=True, volume_cost=1.0, slippage=0.0)["pnl"]
    assert costly <= base
