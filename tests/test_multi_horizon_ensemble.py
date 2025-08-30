import numpy as np

from quant_pipeline.ensemble import SignalEnsemble, MultiHorizonEnsemble


class Dummy:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_multi_horizon_blend():
    ens_short = SignalEnsemble({"m1": Dummy(0.1), "m2": Dummy(-0.2)})
    ens_long = SignalEnsemble({"m1": Dummy(0.3), "m2": Dummy(0.4)})
    mh = MultiHorizonEnsemble({"h1": ens_short, "h5": ens_long})
    signals = {"h1": {"m1": 0.1, "m2": -0.2}, "h5": {"m1": 0.3, "m2": 0.4}}
    weights = {"h1": {"m1": 0.5, "m2": 0.5}, "h5": {"m1": 0.7, "m2": 0.3}}
    blended = mh.blend(signals, weights=weights)
    assert set(blended.keys()) == {"h1", "h5"}
    assert np.isclose(blended["h1"], -0.05)
    assert np.isclose(blended["h5"], 0.33)
