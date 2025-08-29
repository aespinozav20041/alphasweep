import numpy as np
import pandas as pd

from quant_pipeline.features import FeatureBuilder, build_features


def test_feature_builder_matches_batch_and_window():
    bars = [
        {
            "timestamp": 1,
            "close": 100.0,
            "high": 101.0,
            "low": 99.0,
            "bid1": 99.5,
            "ask1": 100.5,
            "bid_sz1": 100,
            "ask_sz1": 120,
            "trades_buy_vol": 50,
            "trades_sell_vol": 30,
        },
        {
            "timestamp": 2,
            "close": 101.0,
            "high": 102.0,
            "low": 100.0,
            "bid1": 100.5,
            "ask1": 101.5,
            "bid_sz1": 110,
            "ask_sz1": 90,
            "trades_buy_vol": 70,
            "trades_sell_vol": 20,
        },
        {
            "timestamp": 3,
            "close": 102.0,
            "high": 103.0,
            "low": 101.0,
            "bid1": 101.5,
            "ask1": 102.5,
            "bid_sz1": 95,
            "ask_sz1": 105,
            "trades_buy_vol": 60,
            "trades_sell_vol": 40,
        },
        {
            "timestamp": 4,
            "close": 103.0,
            "high": 104.0,
            "low": 102.0,
            "bid1": 102.5,
            "ask1": 103.5,
            "bid_sz1": 100,
            "ask_sz1": 100,
            "trades_buy_vol": 80,
            "trades_sell_vol": 60,
        },
        {
            "timestamp": 5,
            "close": 104.0,
            "high": 105.0,
            "low": 103.0,
            "bid1": 103.5,
            "ask1": 104.5,
            "bid_sz1": 120,
            "ask_sz1": 80,
            "trades_buy_vol": 90,
            "trades_sell_vol": 50,
        },
    ]

    df = pd.DataFrame(bars)
    batch = build_features(df)

    fb = FeatureBuilder(seq_len=3)
    frames = [fb.update(bar) for bar in bars]
    stream = pd.concat(frames, ignore_index=True)

    cols = ["ret", "volatility", "spread", "mid_price", "ob_imbalance", "trade_imbalance"]
    assert np.allclose(stream[cols].values, batch[cols].values)

    win = fb.window()
    assert win.shape == (3, len(cols))
    assert np.allclose(win, batch[cols].iloc[-3:].to_numpy())
