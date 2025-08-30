"""Walk-forward evaluation with probability calibration.

This utility performs a rolling IS→OOS split, trains the supplied model,
optionally applies probability calibration for classification tasks and logs
OOS metrics/parameters to :class:`ModelRegistry`.  Parameter sensitivity is
reported as the variance of model parameters across windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from .model_registry import ModelRegistry


@dataclass
class WalkforwardResult:
    """Summary statistics from the walk-forward evaluation."""

    metrics: List[float]
    stability: float


def _calibrate(model, X: np.ndarray, y: np.ndarray, method: str) -> object:
    """Return a probability calibrated version of ``model``.

    Parameters
    ----------
    model : object
        A fitted classifier implementing ``predict_proba``.
    X, y : np.ndarray
        Training data used for calibration.  The estimator is assumed to be
        pre-fit and therefore ``cv='prefit'`` is used.
    method : str
        Calibration method passed to :class:`CalibratedClassifierCV`.  Expected
        values are ``'sigmoid'`` (Platt) or ``'isotonic'``.

    Returns
    -------
    object
        The calibrated model which exposes ``predict_proba``.
    """

    calib = CalibratedClassifierCV(model, method=method, cv="prefit")
    calib.fit(X, y)
    return calib


def _param_stability(params_hist: List[Dict[str, float]]) -> float:
    if not params_hist:
        return 0.0
    keys = params_hist[0].keys()
    vars_ = []
    for k in keys:
        vals = [p[k] for p in params_hist]
        vars_.append(float(np.var(vals)))
    return float(np.mean(vars_))


def walkforward(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_window: int,
    test_window: int,
    step: int,
    train_func: Callable[[np.ndarray, np.ndarray], Tuple[object, Dict[str, float]]],
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    calibrate: Optional[str] = None,
    registry: Optional[ModelRegistry] = None,
    model_id: Optional[int] = None,
) -> WalkforwardResult:
    """Run walk-forward training/testing.

    Parameters
    ----------
    X, y: np.ndarray
        Full dataset features/labels ordered by time.
    train_window, test_window: int
        Number of samples for in-sample (IS) and out-of-sample (OOS) blocks.
    step: int
        Step size to advance the window.
    train_func: Callable
        Function returning (model, params) for provided training data.
    metric_func: Callable
        Computes metric given true labels and predicted probabilities.
    calibrate: Optional[str]
        Either 'sigmoid' for Platt scaling or 'isotonic'.
    registry: ModelRegistry
        Registry to log OOS metrics and parameters.
    model_id: int
        Identifier of model in registry.
    """

    metrics: List[float] = []
    params_hist: List[Dict[str, float]] = []

    n = len(X)
    start = 0
    while start + train_window + test_window <= n:
        end_train = start + train_window
        end_test = end_train + test_window
        X_train, y_train = X[start:end_train], y[start:end_train]
        X_test, y_test = X[end_train:end_test], y[end_train:end_test]

        model, params = train_func(X_train, y_train)
        if calibrate and hasattr(model, "predict_proba"):
            # Only attempt probability calibration for classification models.
            model = _calibrate(model, X_train, y_train, calibrate)
            if registry and model_id is not None:
                prob_true, prob_pred = calibration_curve(
                    y_train, model.predict_proba(X_train)[:, 1], n_bins=10, strategy="quantile"
                )
                registry.log_calibration_curve(
                    model_id, prob_true=prob_true.tolist(), prob_pred=prob_pred.tolist()
                )

        y_prob = model.predict_proba(X_test)[:, 1]
        metric = metric_func(y_test, y_prob)
        metrics.append(metric)
        params_hist.append(params)

        if registry and model_id is not None:
            registry.log_oos_metrics(
                model_id,
                params=params,
                metrics={"metric": metric},
                y_true=y_test,
                y_prob=y_prob,
            )

        start += step

    stability = _param_stability(params_hist)
    return WalkforwardResult(metrics=metrics, stability=stability)


__all__ = ["walkforward", "WalkforwardResult"]
