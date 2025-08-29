import numpy as np

from quant_pipeline.risk import risk_parity_weights, rolling_covariance


def test_risk_parity_weights_sum_to_one():
    cov = np.array([[0.04, 0.006], [0.006, 0.09]])
    w = risk_parity_weights(cov)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w > 0)


def test_rolling_covariance_recent_window():
    returns = np.array([
        [0.01, 0.02],
        [-0.01, 0.03],
        [0.02, -0.02],
        [0.0, 0.01],
    ])
    import pandas as pd

    df = pd.DataFrame(returns, columns=["a", "b"])
    cov = rolling_covariance(df, window=3)
    assert cov.shape == (2, 2)
