import sqlite3

import pandas as pd
import pytest

from pnl_attribution import run


def test_pnl_attribution_run(tmp_path):
    # Create sample trades CSV
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-02"]),
            "model": ["m1", "m1", "m2"],
            "horizon": ["1d", "1d", "1w"],
            "symbol": ["BTC", "ETH", "BTC"],
            "pnl": [1.0, 2.0, -1.0],
            "vol": [0.1, 0.2, 0.3],
            "carry": [0.0, 0.1, 0.0],
            "momo": [-0.1, 0.0, 0.2],
        }
    )
    csv_path = tmp_path / "trades.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    run(csv_path, out_dir)

    # Check CSV outputs
    daily = pd.read_csv(out_dir / "pnl_attribution_daily.csv")
    assert len(daily) == 3
    weekly = pd.read_csv(out_dir / "pnl_attribution_weekly.csv")
    assert len(weekly) == 3
    factors = pd.read_csv(out_dir / "factor_attribution_daily.csv")
    assert factors["vol"].sum() == pytest.approx(0.6)

    # Check SQLite output
    conn = sqlite3.connect(out_dir / "pnl_attribution.db")
    cur = conn.cursor()
    cur.execute("SELECT SUM(pnl) FROM daily_model_attribution")
    total_pnl = cur.fetchone()[0]
    conn.close()
    assert total_pnl == pytest.approx(2.0)

