# Data Table Contracts

## Raw OHLCV
| Column    | Type   | Unit/Description                |
|-----------|--------|---------------------------------|
| timestamp | int64  | milliseconds since Unix epoch UTC |
| open      | float  | price in quote currency (e.g., USD) |
| high      | float  | price in quote currency |
| low       | float  | price in quote currency |
| close     | float  | price in quote currency |
| volume    | float  | traded base-asset units |
| symbol    | string | instrument symbol (e.g., BTCUSDT) |
| timeframe | string | bar size (e.g., 1m, 1h) |

Partitioning: `lake/{exchange}/{symbol}/{timeframe}/YYYY/MM/DD.parquet`

## Feature Table
| Column    | Type  | Unit/Description                |
|-----------|-------|---------------------------------|
| timestamp | int64 | milliseconds since Unix epoch UTC |
| ret       | float | simple return between bars (dimensionless) |
| symbol    | string | instrument symbol |
| timeframe | string | bar size |

Partitioning: same as raw OHLCV.

## News Sentiment Signals
| Column    | Type  | Description                                      |
|-----------|-------|--------------------------------------------------|
| timestamp | int64 | milliseconds since Unix epoch UTC                |
| symbol    | string| instrument symbol                                |
| sentiment | float | normalized news sentiment score in [-1, 1]       |

Partitioning: `lake/news/{symbol}/YYYY/MM/DD.parquet`

## Model Registry (SQLite)
### models
| Column       | Type    | Description                           |
|--------------|---------|---------------------------------------|
| id           | integer | primary key                           |
| ts           | integer | registration timestamp (ms)          |
| type         | text    | model family (xgb, lstm, tcn)        |
| genes_json   | text    | hyperparameter genome (JSON)         |
| artifact_path| text    | path to serialized model artifact    |
| calib_path   | text    | path to calibration data             |
| status       | text    | `champion`, `challenger`, or `retired`|

### model_perf
| Column   | Type    | Description                     |
|----------|---------|---------------------------------|
| model_id | integer | foreign key to models.id        |
| ts       | integer | evaluation timestamp (ms)       |
| ret      | real    | cumulative return over window   |
| sharpe   | real    | Sharpe ratio over window        |
