import numpy as np
from quant_pipeline.drift import DriftDetector


def test_drift_detector(caplog):
    ref = np.linspace(0, 1, 100)
    cur_same = ref.copy()
    detector = DriftDetector(ks_threshold=0.1, psi_threshold=0.1)
    res = detector.check(ref, cur_same)
    assert res.ks == 0.0
    assert res.psi == 0.0

    cur_shift = ref + 1.0
    with caplog.at_level("WARNING"):
        res2 = detector.check(ref, cur_shift)
    assert res2.ks > detector.ks_threshold or res2.psi > detector.psi_threshold
    assert "data drift detected" in caplog.text
