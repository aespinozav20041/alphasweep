import numpy as np

from quant_pipeline.ensemble import SignalEnsemble


def test_blend_multi_horizon_symbol():
    ens = SignalEnsemble({})
    signals = {
        "BTC": {
            "h1": {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])},
            "h2": {"a": np.array([0.5, 0.5]), "b": np.array([-0.5, -0.5])},
        }
    }
    out = ens.blend_multi(
        signals,
        model_weights={"a": 0.5, "b": 0.5},
        horizon_weights={"h1": 0.7, "h2": 0.3},
    )
    assert set(out) == {"BTC"}
    assert out["BTC"].shape == (2,)
