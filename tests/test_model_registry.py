import json
from quant_pipeline.model_registry import ChampionReloader, ModelRegistry


def _mk_files(tmp_path, tag):
    art = tmp_path / f"{tag}.bin"
    calib = tmp_path / f"{tag}.calib"
    art.write_text(tag)
    calib.write_text(tag)
    return str(art), str(calib)


def test_promotion_and_hot_reload(tmp_path):
    db = tmp_path / "reg.db"
    export = tmp_path / "runs/models"
    reg = ModelRegistry(str(db))

    a_art, a_calib = _mk_files(tmp_path, "a")
    champ_id = reg.register_model(
        model_type="ml", genes_json="{}", artifact_path=a_art, calib_path=a_calib, status="champion"
    )
    for i in range(5):
        reg.log_perf(champ_id, ret=0.01, sharpe=1.0, ts=i)

    b_art, b_calib = _mk_files(tmp_path, "b")
    challenger_id = reg.register_model(
        model_type="ml", genes_json="{}", artifact_path=b_art, calib_path=b_calib
    )
    for i in range(5):
        reg.log_perf(challenger_id, ret=0.02, sharpe=1.0, ts=i)

    promoted = reg.evaluate_challengers(
        eval_window_bars=5, uplift_min=0.5, min_bars_to_compare=3, export_dir=str(export)
    )
    assert promoted == challenger_id
    meta = json.loads((export / "current_meta.json").read_text())
    assert meta["id"] == challenger_id

    loads = []

    def loader(path):
        meta = json.loads(path.read_text())
        loads.append(meta["id"])
        return meta["id"]

    reloader = ChampionReloader(export / "current_meta.json", loader)
    assert reloader.poll() == challenger_id

    c_art, c_calib = _mk_files(tmp_path, "c")
    c_id = reg.register_model(
        model_type="ml", genes_json="{}", artifact_path=c_art, calib_path=c_calib
    )
    for i in range(5):
        reg.log_perf(c_id, ret=0.03, sharpe=1.0, ts=i)

    reg.evaluate_challengers(
        eval_window_bars=5, uplift_min=0.2, min_bars_to_compare=3, export_dir=str(export)
    )
    assert reloader.poll() == c_id
    assert loads == [challenger_id, c_id]
