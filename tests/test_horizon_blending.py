import numpy as np
import pandas as pd

from quant_pipeline.ensemble import blend_horizons
from quant_pipeline.labeling import forward_return
from quant_pipeline.model_registry import ModelRegistry
from quant_pipeline.training import AutoTrainer


def test_blend_horizons_weights():
    signals = {1: np.array([1.0, 2.0, 3.0]), 2: np.array([2.0, 2.0, 2.0])}
    blended = blend_horizons(signals)
    # weights: 1 for h=1, 0.5 for h=2 -> normalised 2/3 and 1/3
    expected = np.array([
        1.0 * 2 / 3 + 2.0 * 1 / 3,
        2.0 * 2 / 3 + 2.0 * 1 / 3,
        3.0 * 2 / 3 + 2.0 * 1 / 3,
    ])
    assert np.allclose(blended, expected)


def test_autotrainer_multihorizon_label(tmp_path):
    df = pd.DataFrame({"close": np.linspace(1, 6, 6)})
    reg = ModelRegistry(str(tmp_path / "reg.db"))

    captured = {}

    def loader(_):
        return df

    def train_model(data):
        X, y, _ = data
        captured["y"] = y
        return {}

    trainer = AutoTrainer(
        reg,
        train_every_bars=1,
        history_days=1,
        max_challengers=1,
        dataset_loader=loader,
        train_model=train_model,
        seq_len=2,
        label_horizon=[1, 2],
        label_type="forward_return",
    )

    trainer._train_cycle()

    expected = blend_horizons({
        1: forward_return(df, 1),
        2: forward_return(df, 2),
    })
    assert np.allclose(captured["y"], expected[2:], equal_nan=True)
