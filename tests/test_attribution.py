import pandas as pd
import numpy as np

from src.analytics.pnl_attribution import compute_attribution


def test_attribution_sum() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2025-07-01", periods=5, freq="D"):
        vol = rng.normal(0, 0.5)
        carry = rng.normal(0, 0.2)
        momo = rng.normal(0, 0.3)
        pnl = vol + carry + momo
        rows.append(
            {
                "date": d,
                "model": "m1",
                "strategy": "edge",
                "horizon": "h1",
                "symbol": "AAA",
                "pnl": pnl,
                "vol": vol,
                "carry": carry,
                "momo": momo,
            }
        )
    df = pd.DataFrame(rows)
    res = compute_attribution(df)
    total = df["pnl"].sum()
    assert abs(res.by_component["pnl"].sum() - total) < 1e-9
    assert abs(res.by_factor[["vol", "carry", "momo"]].sum().sum() - total) < 1e-9
