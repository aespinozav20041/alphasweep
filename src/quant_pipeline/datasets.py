"""Dataset utilities for model training."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_lstm_windows(df: pd.DataFrame, seq_len: int):
    """Generate sliding windows for LSTM models.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by time.  Must contain a ``label`` column; all
        other columns are treated as features.
    seq_len : int
        Length of each window.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``X`` with shape ``(n_samples, seq_len, n_features)`` and ``y`` with
        shape ``(n_samples,)``.
    """
    if "label" not in df.columns:
        raise ValueError("missing 'label' column")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if len(df) <= seq_len:
        raise ValueError("df must contain more rows than seq_len")

    feature_cols = [c for c in df.columns if c != "label"]
    X = []
    y = []
    for end in range(seq_len, len(df)):
        start = end - seq_len
        window = df.iloc[start:end][feature_cols].values
        target = df.iloc[end]["label"]
        X.append(window)
        y.append(target)
    return np.asarray(X), np.asarray(y)


__all__ = ["make_lstm_windows"]
