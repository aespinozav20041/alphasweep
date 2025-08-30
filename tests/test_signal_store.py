from quant_pipeline.signal_store import SignalStore


def test_signal_store_roundtrip(tmp_path):
    db = tmp_path / "signals.db"
    store = SignalStore(db)
    store.save(ts=1, strategy="s1", symbol="BTC", horizon="h1", signal=0.5, weight=0.2)
    rows = store.load(strategy="s1", symbol="BTC", horizon="h1", start=0, end=10)
    assert rows == [(1, 0.5, 0.2)]
