import pandas as pd
from quant_pipeline.ingest import combine_market_data


def test_combine_market_data_resample_and_adjustments():
    base = 1_600_000_000_000
    ohlcv = pd.DataFrame(
        [
            [base, 10, 11, 9, 10, 100],
            [base + 30 * 60 * 1000, 11, 12, 10, 11, 150],
        ],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    l2 = pd.DataFrame(
        [[base, 1.0, 2.0], [base + 30 * 60 * 1000, 1.5, 2.5]],
        columns=["timestamp", "bid", "ask"],
    )
    corporate = pd.DataFrame(
        [[base + 15 * 60 * 1000, 2]], columns=["timestamp", "split_ratio"]
    )
    macro = pd.DataFrame(
        [[base, 1], [base + 30 * 60 * 1000, 2]], columns=["timestamp", "macro"]
    )

    combined = combine_market_data(
        ohlcv, l2=l2, corporate=corporate, macro=macro, tz="UTC", resample_rule="1h"
    )

    assert len(combined) == 1
    row = combined.iloc[0]
    assert row.open == 5  # adjusted for split
    assert row.high == 12
    assert row.low == 4.5
    assert row.close == 11
    assert row.volume == 250
    assert row.bid == 1.25 and row.ask == 2.25
    assert row.macro == 2
