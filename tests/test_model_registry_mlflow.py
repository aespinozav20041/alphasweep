import quant_pipeline.model_registry as mr
from quant_pipeline.model_registry import ModelRegistry


def test_log_perf_mlflow(monkeypatch, tmp_path):
    db = tmp_path / "reg.db"
    reg = ModelRegistry(str(db))

    logged = {}

    class DummyMlflow:
        def log_metric(self, name, value, step=None):
            logged[name] = (value, step)

    monkeypatch.setattr(mr, "mlflow", DummyMlflow())

    art = tmp_path / "a.bin"
    calib = tmp_path / "a.calib"
    art.write_text("a")
    calib.write_text("b")
    mid = reg.register_model(
        model_type="ml",
        genes_json="{}",
        artifact_path=str(art),
        calib_path=str(calib),
    )
    reg.log_perf(mid, ret=1.23, sharpe=0.0, ts=5)
    assert logged[f"pnl_model_{mid}"] == (1.23, 5)
