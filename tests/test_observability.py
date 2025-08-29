import re

from fastapi.testclient import TestClient

from quant_pipeline.observability import Observability


def test_metrics_endpoint():
    obs = Observability()
    obs.report_missing_bars("BTC-USDT", "1m", 0.1)
    client = TestClient(obs.app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert re.search("data_missing_bars_ratio", resp.text)
