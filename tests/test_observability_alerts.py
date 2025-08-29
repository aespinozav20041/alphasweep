from quant_pipeline.observability import Observability


def test_observability_alerts(monkeypatch):
    obs = Observability()
    alerts = []
    monkeypatch.setattr(obs, "_send_alert", lambda msg: alerts.append(msg))

    obs.observe_sharpe(0.4, threshold=0.5)
    obs.observe_slippage(12.0, threshold=10.0)
    obs.observe_latency(120.0, threshold=100.0)

    assert any("Sharpe" in a for a in alerts)
    assert any("Slippage" in a for a in alerts)
    assert any("Latency" in a for a in alerts)
