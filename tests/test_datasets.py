import numpy as np
import pandas as pd
from quant_pipeline.datasets import make_lstm_windows
from quant_pipeline.model_registry import ModelRegistry
from quant_pipeline.training import AutoTrainer


def _dummy_loader(_):
    n = 30
    return pd.DataFrame({
        "feat": np.arange(n, dtype=float),
        "label": np.arange(n, dtype=float),
    })


def _dummy_train(_):
    return {}


def test_make_lstm_windows_shapes():
    df = _dummy_loader(0)
    X, y = make_lstm_windows(df, seq_len=5)
    assert X.shape == (25, 5, 1)
    assert y.shape == (25,)


def test_autotrainer_build_dataset_no_leakage(tmp_path):
    reg = ModelRegistry(str(tmp_path / "db.db"))
    trainer = AutoTrainer(
        reg,
        train_every_bars=1,
        history_days=5,
        max_challengers=1,
        dataset_loader=_dummy_loader,
        train_model=_dummy_train,
        seq_len=5,
        cv_splits=3,
        embargo=2,
    )
    X, y, splits = trainer.build_dataset(_dummy_loader(0))
    assert X.shape == (25, 5, 1)
    assert y.shape == (25,)
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))
        for ti in test_idx:
            assert all(abs(ti - tr) > 2 for tr in train_idx)
