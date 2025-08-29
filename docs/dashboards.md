# Trading Dashboards

The system exposes trading metrics via Prometheus to build dashboards for:

- **PnL**
- **Exposure**
- **Hit-rate**
- **Drawdown**
- **Fills vs orders**
- **Latencies**

Use `trading.metrics.start_metrics_server` to start the HTTP endpoint and update the
metrics during execution. Metrics can then be scraped by Prometheus and visualised
in tools like Grafana.
