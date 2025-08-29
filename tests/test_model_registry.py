import json
from quant_pipeline.model_registry import ChampionReloader, ModelRegistry


def _mk_files(tmp_path, tag):
    files = {}
    for suffix in ["bin", "calib", "lstm", "scaler", "feat", "thresh", "risk"]:
        p = tmp_path / f"{tag}.{suffix}"
        p.write_text(tag)
        files[suffix] = str(p)
    return files


def test_promotion_and_hot_reload(tmp_path):
    db = tmp_path / "reg.db"
    export = tmp_path / "runs/models"
    reg = ModelRegistry(str(db))

    a_files = _mk_files(tmp_path, "a")
    champ_id = reg.register_model(
        model_type="ml",
        genes_json="{}",
        artifact_path=a_files["bin"],
        calib_path=a_files["calib"],
        lstm_path=a_files["lstm"],
        scaler_path=a_files["scaler"],
        features_path=a_files["feat"],
        thresholds_path=a_files["thresh"],
        risk_rules_path=a_files["risk"],
        ga_version="v1",
        seed=1,
        data_hash="hash-a",
        status="champion",
    )
    for i in range(5):
        reg.log_perf(champ_id, ret=0.01, sharpe=1.0, ts=i)

    b_files = _mk_files(tmp_path, "b")
    challenger_id = reg.register_model(
        model_type="ml",
        genes_json="{}",
        artifact_path=b_files["bin"],
        calib_path=b_files["calib"],
        lstm_path=b_files["lstm"],
        scaler_path=b_files["scaler"],
        features_path=b_files["feat"],
        thresholds_path=b_files["thresh"],
        risk_rules_path=b_files["risk"],
        ga_version="v1",
        seed=1,
        data_hash="hash-b",
    )
    for i in range(5):
        reg.log_perf(challenger_id, ret=0.02, sharpe=1.0, ts=i)

    promoted = reg.evaluate_challengers(
        eval_window_bars=5,
        uplift_min=0.5,
        min_bars_to_compare=3,
        export_dir=str(export),
    )
    assert promoted == challenger_id
    meta = json.loads((export / "current_meta.json").read_text())
    assert meta["id"] == challenger_id
    assert meta["ga_version"] == "v1"
    assert meta["seed"] == 1
    assert meta["data_hash"] == "hash-b"
    assert meta["risk_rules"]
    assert (export / "current_lstm").read_text() == "b"
    assert (export / "current_scaler").read_text() == "b"
    assert (export / "current_features").read_text() == "b"
    assert (export / "current_thresholds").read_text() == "b"
    assert (export / "current_risk_rules").read_text() == "b"

    loads = []

    def loader(path):
        meta = json.loads(path.read_text())
        loads.append(meta["id"])
        return meta["id"]

    reloader = ChampionReloader(export / "current_meta.json", loader)
    assert reloader.poll() == challenger_id

    c_files = _mk_files(tmp_path, "c")
    c_id = reg.register_model(
        model_type="ml",
        genes_json="{}",
        artifact_path=c_files["bin"],
        calib_path=c_files["calib"],
        lstm_path=c_files["lstm"],
        scaler_path=c_files["scaler"],
        features_path=c_files["feat"],
        thresholds_path=c_files["thresh"],
        risk_rules_path=c_files["risk"],
        ga_version="v1",
        seed=1,
        data_hash="hash-c",
    )
    for i in range(5):
        reg.log_perf(c_id, ret=0.03, sharpe=1.0, ts=i)

    reg.evaluate_challengers(
        eval_window_bars=5,
        uplift_min=0.2,
        min_bars_to_compare=3,
        export_dir=str(export),
    )
    assert reloader.poll() == c_id
    assert loads == [challenger_id, c_id]

