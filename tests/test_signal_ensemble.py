import numpy as np

from quant_pipeline.ensemble import SignalEnsemble


def test_blend_combines_signals():
    ens = SignalEnsemble({})
    blended = ens.blend(
        {"a": np.array([1.0, -1.0]), "b": np.array([0.0, 1.0])},
        weights={"a": 0.5, "b": 0.5},
    )
    assert np.allclose(blended, np.array([0.5, 0.0]))
