import numpy as np
import pandas as pd

from quant_pipeline.features import (
    build_features,
    FeatureNormalizer,
    generate_lstm_windows,
)


def make_df(rows: int = 10):
    """Create a deterministic OHLCV DataFrame."""

    return pd.DataFrame(
        {
            "timestamp": range(rows),
            "open": np.linspace(1.0, 1.5, rows),
            "high": np.linspace(1.1, 1.6, rows),
            "low": np.linspace(0.9, 0.4, rows),
            "close": np.linspace(1.0, 1.3, rows),
            "volume": np.arange(1, rows + 1, dtype=float),
        }
    )


def test_build_features_indicators():
    df = make_df()
    feats = build_features(df)
    for col in {"ret", "volatility", "spread", "price_impact"}:
        assert col in feats.columns


def test_normalizer_fit_and_ffill():
    df = pd.DataFrame({"ret": [1.0, np.nan, np.nan, 2.0]})
    train = df.iloc[[0, 3]]
    norm = FeatureNormalizer(["ret"], ffill_limit=1)
    norm.fit(train)
    transformed = norm.transform(df)
    # second row should be forward filled, third should remain NaN
    assert np.isclose(transformed.loc[1, "ret"], -0.70710677, atol=1e-6)
    assert pd.isna(transformed.loc[2, "ret"])


def test_generate_lstm_windows_shape():
    df = make_df()
    feats = build_features(df).dropna()
    norm = FeatureNormalizer(["ret", "volatility"], ffill_limit=1)
    norm.fit(feats)
    normed = norm.transform(feats)
    windows = generate_lstm_windows(
        normed, seq_len=3, feature_cols=["ret", "volatility"]
    )
    assert windows.shape == (len(normed) - 3 + 1, 3, 2)
    # first window's first feature sequence should match original values
    assert np.allclose(windows[0, :, 0], normed["ret"].to_numpy()[:3])
