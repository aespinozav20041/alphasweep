# alphasweep

This repository contains a minimalist quantitative trading pipeline. See the
[docs](docs/) for:
- [Data table contracts](docs/contracts.md)
- [Data quality playbooks](docs/playbooks.md)
- [Data source catalog](docs/catalog.md)
- [Data Doctor before/after examples](docs/data_doctor.md)

## Training

The training pipeline can automatically derive meta-labels using the triple
barrier method.  Provide labeling parameters to :class:`AutoTrainer` and the
dataset passed to ``train_model`` will include a ``label`` column:

```python
from quant_pipeline.training import AutoTrainer

trainer = AutoTrainer(
    registry,
    train_every_bars=100,
    history_days=5,
    max_challengers=3,
    build_dataset=load_recent_data,
    train_model=train_model,
    label_horizon=5,
    label_up_mult=2.0,
    label_down_mult=1.5,
)
```

The underlying labeling logic is exposed as
``quant_pipeline.labels.triple_barrier`` for standalone use.
