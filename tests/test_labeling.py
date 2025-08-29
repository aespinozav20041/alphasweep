import pandas as pd
from quant_pipeline.labeling import forward_return, triple_barrier_labels


def test_forward_return_simple():
    df = pd.DataFrame({"close": [1.0, 1.1, 1.21]})
    result = forward_return(df, 1)
    expected = pd.Series([0.1, 0.1, float("nan")], name="label")
    pd.testing.assert_series_equal(result, expected)


def test_triple_barrier_labels():
    df = pd.DataFrame(
        {
            "close": [100, 100, 100, 100],
            "high": [100, 103, 100, 100],
            "low": [100, 100, 97, 100],
        }
    )
    labels = triple_barrier_labels(df, upper=0.02, lower=0.02, horizon=2)
    expected = pd.Series([1, -1, 0, 0], name="label")
    pd.testing.assert_series_equal(labels, expected)
