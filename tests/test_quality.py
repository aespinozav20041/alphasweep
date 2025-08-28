import json
from pathlib import Path

import pandas as pd

from quant_pipeline.models import BarOHLCV
from quant_pipeline.quality import (
    trading_calendar,
    find_missing_bars,
    repair_missing_bars,
    clip_outliers,
    validate_contract,
    doctor_ohlcv,
)
from quant_pipeline.storage import to_parquet, read_table


def sample_df(start: int, tf_ms: int):
    ts = [start, start + tf_ms, start + 3 * tf_ms]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["BTC/USDT"] * 3,
            "open": [1.0, 2.0, 4.0],
            "high": [1.0, 2.0, 4.0],
            "low": [1.0, 2.0, 4.0],
            "close": [1.5, 2.5, 4.5],
            "volume": [10.0, 20.0, 40.0],
            "source": ["test"] * 3,
            "timeframe": ["1h"] * 3,
        }
    )


def test_find_and_repair_missing():
    start = 0
    tf_ms = 3600_000
    end = start + 3 * tf_ms
    df = sample_df(start, tf_ms)
    cal = trading_calendar(start, end, "1h")
    missing = find_missing_bars(df, cal)
    assert missing["gap_len"].sum() == 1
    repaired = repair_missing_bars(df, cal)
    assert len(repaired) == len(cal)
    gap_ts = start + 2 * tf_ms
    row = repaired[repaired["timestamp"] == gap_ts].iloc[0]
    assert row.open == row.close == 2.5
    assert row.volume == 0.0


def test_clip_and_validate():
    start = 0
    tf_ms = 3600_000
    end = start + 3 * tf_ms
    df = sample_df(start, tf_ms)
    df.loc[2, "volume"] = 1_000_000.0
    clipped, cnt = clip_outliers(df, price_z=1.0, vol_z=1.0)
    assert cnt > 0
    report = validate_contract(clipped, BarOHLCV, outlier_count=cnt)
    assert report.rows == len(clipped)
    assert report.outlier_count == cnt
    assert report.dup_ratio == 0.0


def test_doctor_pipeline(tmp_path):
    lake = tmp_path / "lake"
    runs = tmp_path / "runs"
    start = 0
    tf_ms = 3600_000
    end = start + 3 * tf_ms
    raw = sample_df(start, tf_ms)
    to_parquet(raw, "ohlcv", base_path=lake)

    report = doctor_ohlcv(
        "BTC/USDT",
        "1h",
        start,
        end,
        gap_threshold=0.5,
        lake_path=lake,
        runs_path=runs,
    )
    assert report.missing_bars == 1

    clean = read_table("ohlcv_clean", ["BTC/USDT"], start, end, "1h", base_path=lake)
    assert len(clean) == 4
    gap_ts = start + 2 * tf_ms
    row = clean[clean["timestamp"] == gap_ts].iloc[0]
    assert row.volume == 0.0

    rep_file = runs / "reports" / "quality"
    files = list(rep_file.glob("*.json"))
    assert files and json.loads(files[0].read_text())["missing_bars"] == 1
