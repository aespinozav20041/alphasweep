import numpy as np
import pandas as pd

from quant_pipeline.adapters import XGBAdapter, TCNAdapter, RuleStrategy


def test_xgb_adapter_basic():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    model = XGBAdapter()
    model.fit(X, y)
    pred = model.predict(X[:2])
    assert pred.shape == (2,)


def test_tcn_adapter_basic():
    X = np.random.randn(20, 5, 1)
    y = np.random.randn(20)
    model = TCNAdapter(input_size=1)
    model.fit(X, y, epochs=1)
    pred = model.predict(X[:2])
    assert pred.shape == (2,)


def test_rule_strategy():
    df = pd.DataFrame({"a": [-1, 2, -3, 4]})
    strat = RuleStrategy(threshold=0.0)
    sig = strat.predict(df)
    assert (sig == np.array([-1, 1, -1, 1])).all()
