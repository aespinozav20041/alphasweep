# Runtime Options

The `DecisionLoop` provides optional post-processing utilities to smooth signals and reduce churn.

```python
from quant_pipeline.decision import DecisionLoop

loop = DecisionLoop(
    model,
    risk,
    oms,
    obs,
    median_window=5,
    hysteresis=0.05,
)
```

- `median_window` applies a median filter of the given window size to the EMA output.
- `hysteresis` adds symmetric entry/exit bands around the `threshold` to avoid rapid toggling.
