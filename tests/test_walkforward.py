import json
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from quant_pipeline.walkforward import walkforward
from quant_pipeline.model_registry import ModelRegistry


def test_walkforward_logs_oos_and_stability(tmp_path):
    X, y = make_classification(n_samples=120, n_features=5, random_state=0)
    reg = ModelRegistry(str(tmp_path / "reg.db"))
    model_id = reg.register_model(
        model_type="lr", genes_json="{}", artifact_path="a", calib_path="c"
    )

    def train_func(X_tr, y_tr):
        model = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
        params = {"coef": float(model.coef_[0][0])}
        return model, params

    def metric_fn(y_true, y_prob):
        preds = (y_prob > 0.5).astype(int)
        return float((preds == y_true).mean())

    result = walkforward(
        X,
        y,
        train_window=60,
        test_window=20,
        step=20,
        train_func=train_func,
        metric_func=metric_fn,
        calibrate="sigmoid",
        registry=reg,
        model_id=model_id,
    )

    assert len(result.metrics) == 3
    assert result.stability >= 0.0

    rows = reg.list_oos_metrics(model_id)
    assert len(rows) == 3
    first = rows[0]
    params = json.loads(first["params_json"])
    metrics = json.loads(first["metrics_json"])
    assert "coef" in params
    assert "metric" in metrics


    calib_rows = reg.list_calibration_curves(model_id)
    assert len(calib_rows) == 3
=======
    assert "calibration_curve" in metrics
    for k in ["calibration_json", "calibration_csv", "calibration_png"]:
        p = Path(metrics[k])
        assert p.exists()

