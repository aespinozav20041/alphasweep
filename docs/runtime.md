# Runtime Options

The `DecisionLoop` provides optional post-processing utilities to smooth signals and reduce churn.

```python
from quant_pipeline.decision import DecisionLoop

loop = DecisionLoop(
    model,
    risk,
    oms,
    obs,
    meta_model=meta_clf,
    p_long=0.6,
    p_short=0.6,
    median_window=5,
    hysteresis=0.05,
)
```

- `median_window` applies a median filter of the given window size to the EMA output.
- `hysteresis` adds symmetric entry/exit bands around the `threshold` to avoid rapid toggling.
- `meta_model` provides meta-labeling probabilities; trades are only executed when
  they exceed `p_long`/`p_short` and the calibrated probability drives sizing.
