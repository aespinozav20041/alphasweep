import numpy as np
import pandas as pd

from quant_pipeline.features import normalize_train_only, sliding_window_tensor


def test_normalize_and_sliding_windows():
    data = pd.DataFrame(
        {
            "feat1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "feat2": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )
    train, test = data.iloc[:3], data.iloc[3:]
    train_scaled, test_scaled, _ = normalize_train_only(
        train, test, columns=["feat1", "feat2"], ffill_limit=1
    )
    # train mean should be ~0 and std ~1 for scaled columns
    assert np.allclose(train_scaled[["feat1", "feat2"]].mean(), 0.0)
    assert np.allclose(train_scaled[["feat1", "feat2"]].std(ddof=0), 1.0)
    assert not train_scaled.isna().any().any()

    tensor = sliding_window_tensor(train_scaled, 2, ["feat1", "feat2"])
    assert tensor.shape == (len(train_scaled) - 2 + 1, 2, 2)
