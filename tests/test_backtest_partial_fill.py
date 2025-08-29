import pandas as pd

from quant_pipeline.backtest import run_backtest


def test_partial_fills_reduce_turnover():
    df = pd.DataFrame({"ret": [0.01, -0.02, 0.03, -0.01], "volume": [1, 1, 1, 1]})
    full = run_backtest(df, return_metrics=True)
    partial = run_backtest(df, return_metrics=True, max_volume_frac=0.25)
    assert partial["turnover"] <= full["turnover"]
