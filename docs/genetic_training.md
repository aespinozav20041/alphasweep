# Genetic hyper-parameter training

The function `train_with_genetic` uses `GeneticOptimizer` to search over
model and post-processing parameters:

- `n_lstm`
- `hidden`
- `dropout`
- `seq_len`
- `horizon`
- `lr`
- `weight_decay`
- `umbral`
- `ema`
- `cooldown`

When combined with :class:`~quant_pipeline.training.AutoTrainer`, the registry
automatically stores the selected genes and their out-of-sample metrics.

```python
import pandas as pd
from quant_pipeline.model_registry import ModelRegistry
from quant_pipeline.training import AutoTrainer, train_with_genetic

# toy dataset containing sequential returns
df = pd.DataFrame({"ret": [0.01, -0.02, 0.03, -0.01]})

# AutoTrainer will call `train_with_genetic` and record the resulting model
reg = ModelRegistry("reg.db")
trainer = AutoTrainer(
    reg,
    train_every_bars=1,
    history_days=1,
    max_challengers=1,
    build_dataset=lambda _: df,
    train_model=train_with_genetic,
)

trainer.start()
trainer.notify_bar()  # triggers a training cycle
trainer.stop()
```

The best genes are returned in ``genes_json`` and the full parameter/metric
record is logged via :meth:`ModelRegistry.log_oos_metrics`.
