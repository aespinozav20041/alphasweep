import time
import pandas as pd

from quant_pipeline.model_registry import ModelRegistry
from quant_pipeline.training import AutoTrainer, train_with_genetic


def test_genetic_training_logs_metrics(tmp_path):
    df = pd.DataFrame({"ret": [0.01, -0.02, 0.03, -0.01]})
    reg = ModelRegistry(str(tmp_path / "reg.db"))

    trainer = AutoTrainer(
        reg,
        train_every_bars=1,
        history_days=1,
        max_challengers=1,
        build_dataset=lambda _: df,
        train_model=train_with_genetic,
    )

    trainer.start()
    trainer.notify_bar()

    model_id = None
    for _ in range(50):
        models = reg.list_models(status="challenger")
        if models:
            model_id = models[0]["id"]
            break
        time.sleep(0.05)

    trainer.stop()

    assert model_id is not None
    oos = reg.list_oos_metrics(model_id)
    assert len(oos) == 1
    assert "params_json" in oos[0] and "metrics_json" in oos[0]
