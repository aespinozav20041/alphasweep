import numpy as np
import pandas as pd
import pytest

from quant_pipeline.storage import to_parquet, read_table
from quant_pipeline.doctor import validate_ohlcv
from quant_pipeline.features import build_features
from quant_pipeline.backtest import run_backtest


def synth_ohlcv(start: int, bars: int, tf_ms: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 100.0
    rows = []
    for i in range(bars):
        open_ = price
        close = open_ * (1 + rng.normal(0, 0.001))
        high = max(open_, close) * (1 + rng.random() * 0.001)
        low = min(open_, close) * (1 - rng.random() * 0.001)
        volume = float(rng.random())
        ts = start + i * tf_ms
        rows.append(
            dict(
                timestamp=ts,
                symbol="BTC-USDT",
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                source="sim",
                timeframe="1m",
            )
        )
        price = close
    return pd.DataFrame(rows)


def test_e2e_pipeline(tmp_path):
    tf_ms = 60_000
    start = 1_600_000_000_000
    df = synth_ohlcv(start, 50, tf_ms, seed=1)
    to_parquet(df, "ohlcv", base_path=tmp_path)
    loaded = read_table(
        "ohlcv",
        symbols=["BTC-USDT"],
        start=start,
        end=start + 49 * tf_ms,
        timeframe="1m",
        base_path=tmp_path,
    )
    checked = validate_ohlcv(loaded, tf_ms)
    feats = build_features(checked)
    pnl = run_backtest(feats)
    assert isinstance(pnl, float)


def test_doctor_rejects_gaps(tmp_path):
    tf_ms = 60_000
    start = 1_600_000_000_000
    df = synth_ohlcv(start, 10, tf_ms, seed=2)
    df.loc[5, "timestamp"] += tf_ms * 5  # create large gap
    df = df.sort_values("timestamp").reset_index(drop=True)
    to_parquet(df, "ohlcv", base_path=tmp_path)
    loaded = read_table(
        "ohlcv",
        symbols=["BTC-USDT"],
        start=start,
        end=start + 15 * tf_ms,
        timeframe="1m",
        base_path=tmp_path,
    )
    with pytest.raises(ValueError):
        validate_ohlcv(loaded, tf_ms)
