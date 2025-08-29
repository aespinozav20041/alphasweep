import pandas as pd

from quant_pipeline.utils import resample_ohlcv


def test_resample_ohlcv_detects_split():
    base = 1_600_000_000_000
    tf = 3_600_000
    df = pd.DataFrame(
        [
            [base, 100.0, 110.0, 90.0, 110.0, 100],
            [base + tf, 55.0, 56.0, 54.0, 55.0, 200],
        ],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    out = resample_ohlcv(df, "1h")
    first = out.iloc[0]
    assert first.open == 50.0
    assert first.close == 55.0


def test_resample_ohlcv_aligns_timeframe():
    base = 1_600_000_000_000
    tf = 3_600_000
    df = pd.DataFrame(
        [
            [base + 1_000, 1.0, 2.0, 0.5, 1.5, 10],
            [base + tf + 1_000, 1.6, 2.6, 1.1, 2.0, 20],
        ],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    out = resample_ohlcv(df, "1h")
    expected_start = base - (base % tf)
    assert list(out["timestamp"]) == [expected_start, expected_start + tf]
    diffs = out["timestamp"].diff().dropna()
    assert (diffs == tf).all()
