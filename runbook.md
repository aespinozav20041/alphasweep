# Runbook

## Canary deployment

Run the system in "canary" mode with extremely low exposure.

```bash
./bin/canary.sh
```

This script sets:

- `MAX_NOTIONAL_PER_ORDER=50`
- `MAX_TOTAL_NOTIONAL=200`
- `MAX_DD_DAILY=0.03`
- `LATENCY_THRESHOLD_MS=500`

These values cap notional per order and total exposure, trigger the kill-switch at 3% daily drawdown, and pause if latency exceeds 500 ms.

## Pre-live checklist

1. **Data** – verify recent OHLCV is present and no missing bars: `python -m quant_pipeline ingest-ohlcv --exchange binance --symbols BTC/USDT`.
2. **Metrics** – ensure Prometheus endpoint responds: `curl localhost:8000/metrics`.
3. **Alerts** – send a test alert via `python -m quant_pipeline.observability --test-alert` and confirm receipt.
4. **OMS** – dry-run an order through the OMS: `PYTHONPATH=src python -m tests.test_oms`.
5. **Kill-switch** – simulate a drawdown to ensure pause: `PYTHONPATH=src python - <<'PY'
from quant_pipeline.risk import RiskManager
r = RiskManager(max_dd_daily=0.03,max_dd_weekly=0.05,latency_threshold=500,latency_window=3,pause_minutes=5)
r.update_drawdowns(0.03,0)
print('paused', r.is_paused())
PY`.

Do not go live until all items pass.

## Rollback plan

If issues arise:

1. **Pause trading loop**
   ```bash
   kill -SIGINT $(pgrep -f quant_pipeline)
   ```
2. **Close all positions**
   ```bash
   PYTHONPATH=src python - <<'PY'
from quant_pipeline.oms import OMS
oms = OMS()
oms.close_all()
PY
   ```
3. **Restore previous champion model**
   ```bash
   PYTHONPATH=src python - <<'PY'
from quant_pipeline.model_registry import ModelRegistry
reg = ModelRegistry('runs/models.db')
reg.promote_model(<prev_id>, 'runs/models')
PY
   ```

## Escalation rules

Increase trade size only when rolling 30 d Sharpe > 0 and no kill-switch events for 7 d:

- Double `MAX_NOTIONAL_PER_ORDER` weekly until target size.
- Scale `MAX_TOTAL_NOTIONAL` proportionally.

Monitor drawdowns, latency and alerts continuously while scaling.

## Alert diagnostics

Use Slack or email alerts as a trigger to investigate and roll back if
necessary.

- **Connection failure**
  1. Verify network reachability to the affected service.
  2. Check service logs and restart the component or switch to a backup.
  3. If connectivity cannot be restored quickly, pause trading and follow
     the rollback plan.
- **Timeout**
  1. Identify the operation from the alert message.
  2. Inspect `/metrics` for latency anomalies.
  3. Retry once; if timeouts persist, pause trading and roll back.
- **Anomalous ratios** (e.g. `data_missing_bars_ratio`,
  `risk_fetch_failures_total`)
  1. Confirm the metric on `/metrics`.
  2. Run data or risk diagnostics scripts to locate the source.
  3. If the ratio remains elevated, execute the rollback plan.

## Alert rollback

Unresolved alerts require rolling back:

1. Pause the trading loop.
2. Close outstanding positions.
3. Restore the previous champion model.
