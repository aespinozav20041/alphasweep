# Data Doctor Examples

## Before
```
timestamp,open,high,low,close,volume
1000,1.0,1.1,0.9,1.05,10
1000,1.05,1.2,1.0,1.10,12  # duplicate
4000,1.15,1.3,1.1,1.25,10  # gap at 2000-3000
```

## After running `validate_ohlcv`
```
timestamp,open,high,low,close,volume
1000,1.0,1.1,0.9,1.05,10
2000,1.1,1.2,1.0,1.15,11
3000,1.15,1.3,1.1,1.25,10
4000,1.25,1.35,1.2,1.30,9
```

```
from quant_pipeline.doctor import validate_ohlcv
import pandas as pd

df = pd.read_csv('raw.csv')
clean = validate_ohlcv(df, timeframe_ms=1000)
clean.to_csv('clean.csv', index=False)
```
