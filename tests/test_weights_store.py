from quant_pipeline.weights_store import save_weights, load_weights


def test_save_and_load_weights(tmp_path):
    path = tmp_path / "w.db"
    weights = {"strat1": {"BTC": {"h1": 0.5}}}
    save_weights(path, weights)
    loaded = load_weights(path)
    assert loaded == weights
