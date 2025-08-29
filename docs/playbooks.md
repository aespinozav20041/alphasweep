# Data Quality Playbooks

## Fixing Missing Bars or Outliers
1. Run the doctor on the suspect range to identify issues:
   ```bash
   python - <<'PY'
from pathlib import Path
import pandas as pd
from quant_pipeline import doctor

df = pd.read_csv('lake/BTCUSDT/1m/2024/01/01.csv')
try:
    doctor.validate_ohlcv(df, timeframe_ms=60_000)
except ValueError as exc:
    print(exc)
PY
   ```
2. Re-ingest data from the exchange for the flagged window.
3. Overwrite the affected files in `lake/...` with the fresh dump.
4. Re-run the doctor to confirm the gap/outlier is gone.

## Re-processing a Date Range
1. Remove existing feature and model artifacts for the date span:
   ```bash
   rm -rf data/features/2024-01-{01..07}
   ```
2. Rebuild features and retrain models:
   ```bash
   PYTHONPATH=src python -m quant_pipeline.training --start 2024-01-01 --end 2024-01-07
   ```
3. Validate the new outputs with the doctor and backtester before promotion.
