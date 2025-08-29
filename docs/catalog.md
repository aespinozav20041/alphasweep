# Data Source Catalog

| Source   | Endpoint Type | Rate Limit          | Typical Latency |
|----------|---------------|--------------------|-----------------|
| Binance  | REST API      | 1200 req/min        | 150-250 ms      |
| Coinbase | REST API      | 10 req/sec          | 200-300 ms      |
| Kraken   | REST API      | 20 req/sec          | 250-350 ms      |
| Binance  | WebSocket     | 1 stream / conn     | <100 ms         |

Notes:
- Batch historical downloads during off-peak hours to respect rate limits.
- Monitor latency via the `/metrics` endpoint; sustained spikes trigger alerts.
