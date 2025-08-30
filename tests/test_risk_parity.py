import numpy as np
import pandas as pd

from quant_pipeline.risk_parity import risk_parity_weights, rolling_risk_parity


def test_risk_parity_weights_basic():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], columns=["A", "B"], index=["A", "B"])
    w = risk_parity_weights(cov)
    assert np.allclose(w.values, [0.6, 0.4], atol=1e-6)


def test_rolling_risk_parity():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(20, 2))
    rets = pd.DataFrame(data, columns=["A", "B"])
    weights = rolling_risk_parity(rets, window=5)
    assert weights.shape[0] == 16
    assert list(weights.columns) == ["A", "B"]

