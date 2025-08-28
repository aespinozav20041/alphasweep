import pandas as pd
import pytest

from quant_pipeline.storage import to_parquet, read_table


def sample_bars():
    return pd.DataFrame(
        [
            {
                "timestamp": 1_600_000_000_000,
                "symbol": "BTC-USDT",
                "open": 10.0,
                "high": 12.0,
                "low": 9.0,
                "close": 11.0,
                "volume": 1.0,
                "source": "binance",
                "timeframe": "1m",
            },
            {
                "timestamp": 1_600_000_060_000,
                "symbol": "BTC-USDT",
                "open": 11.0,
                "high": 13.0,
                "low": 10.0,
                "close": 12.0,
                "volume": 2.0,
                "source": "binance",
                "timeframe": "1m",
            },
        ]
    )


def test_roundtrip(tmp_path):
    df = sample_bars()
    to_parquet(df, "ohlcv", base_path=tmp_path)
    out = read_table(
        "ohlcv",

=======
=======
    to_parquet(df, "bar_ohlcv", base_path=tmp_path)
    out = read_table(
        "bar_ohlcv",

        symbols=["BTC-USDT"],
        start=df.timestamp.min(),
        end=df.timestamp.max(),
        timeframe="1m",
        base_path=tmp_path,
    )
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df)


def test_missing_column(tmp_path):
    df = sample_bars().drop(columns=["close"])
    with pytest.raises(ValueError):
        to_parquet(df, "ohlcv", base_path=tmp_path)

=======
=======
        to_parquet(df, "bar_ohlcv", base_path=tmp_path)


def test_volume_default(tmp_path):
    df = sample_bars()
    df.loc[0, "volume"] = float("nan")

    to_parquet(df, "ohlcv", base_path=tmp_path)
    out = read_table(
        "ohlcv",
=======

    to_parquet(df, "ohlcv", base_path=tmp_path)
    out = read_table(
        "ohlcv",
=======
    to_parquet(df, "bar_ohlcv", base_path=tmp_path)
    out = read_table(
        "bar_ohlcv",

        symbols=["BTC-USDT"],
        start=df.timestamp.min(),
        end=df.timestamp.max(),
        timeframe="1m",
        base_path=tmp_path,
    )
    assert out.loc[0, "volume"] == 0.0
