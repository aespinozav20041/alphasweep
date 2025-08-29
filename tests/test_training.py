import time
import numpy as np
import pandas as pd
from quant_pipeline.model_registry import ModelRegistry
from quant_pipeline.training import AutoTrainer


def _mk_files(tmp_path, tag):
    art = tmp_path / f"{tag}.bin"
    calib = tmp_path / f"{tag}.calib"
    art.write_text(tag)
    calib.write_text(tag)
    return str(art), str(calib)


def test_autotrainer_registers_and_prunes(tmp_path):
    db = tmp_path / "reg.db"
    reg = ModelRegistry(str(db))

    trained = []

    def dataset_loader(days):
        n = 10
        df = pd.DataFrame({
            "feat": np.arange(n, dtype=float),
            "label": np.arange(n, dtype=float),
        })
        return df

    def train_model(_):
        idx = len(trained)
        art, calib = _mk_files(tmp_path, f"m{idx}")
        trained.append(idx)
        return {
            "type": "xgb",
            "genes_json": "{}",
            "artifact_path": art,
            "calib_path": calib,
        }

    trainer = AutoTrainer(
        reg,
        train_every_bars=1,
        history_days=3,
        max_challengers=2,
        dataset_loader=dataset_loader,
        train_model=train_model,
        seq_len=2,
        cv_splits=2,
    )

    trainer.start()
    trainer.notify_bar()
    for _ in range(50):
        if len(reg.list_models(status="challenger")) == 1:
            first_id = reg.list_models(status="challenger")[0]["id"]
            break
        time.sleep(0.05)

    trainer.notify_bar()
    for _ in range(50):
        if len(reg.list_models(status="challenger")) == 2:
            break
        time.sleep(0.05)

    trainer.notify_bar()
    for _ in range(50):
        ids = [m["id"] for m in reg.list_models(status="challenger")]
        if len(ids) == 2 and first_id not in ids:
            break
        time.sleep(0.05)

    trainer.stop()

    challengers = reg.list_models(status="challenger")
    assert len(challengers) == 2
    ids = [c["id"] for c in challengers]
    assert first_id not in ids
    assert trained == [0, 1, 2]
