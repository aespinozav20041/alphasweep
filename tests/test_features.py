import numpy as np
import pandas as pd
import pytest

from quant_pipeline.features import build_features, causal_normalize, sliding_window_tensor


def _basic_df():
    return pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "volume": [1, 1, 1, 1],
        }
    )


def test_build_features_creates_indicators():
    feats = build_features(_basic_df(), vol_window=2)
    for col in ["ret", "vol", "hl_spread", "oc_spread", "pos_in_range"]:
        assert col in feats.columns
    assert feats["vol"].iloc[0] == 0.0
    assert feats["hl_spread"].iloc[0] == pytest.approx((1.1 - 0.9) / 1.05)


def test_causal_normalize_forward_fill():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
    norm = causal_normalize(df, limit=1)
    assert norm.shape == df.shape
    # second row uses first value for mean -> (2 - 1) / std where std=1
    assert norm.iloc[1, 0] == pytest.approx(1.0)
    # third row forward fills 2.0 with limit=1
    expected = (2.0 - 1.5) / np.std([1.0, 2.0], ddof=1)
    assert norm.iloc[2, 0] == pytest.approx(expected)


def test_sliding_window_tensor_shape():
    feats = build_features(_basic_df())[["ret", "vol"]]
    tensor = sliding_window_tensor(feats, seq_len=2, stride=1)
    assert tensor.shape == (3, 2, 2)
    assert np.array_equal(tensor[0], feats.iloc[0:2].to_numpy())
