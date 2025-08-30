import numpy as np

from quant_pipeline.model_registry import plot_calibration


def test_plot_calibration(tmp_path):
    probs = np.linspace(0.1, 0.9, 5)
    empir = probs ** 0.5
    out = tmp_path / "cal.png"
    plot_calibration(probs, empir, path=out)
    assert out.exists()

