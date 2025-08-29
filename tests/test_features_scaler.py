from quant_pipeline.features import FeatureBuilder, Scaler


def test_feature_builder_and_scaler_stateful():
    fb = FeatureBuilder()
    sc = Scaler()
    bar1 = {"timestamp": 1, "close": 100.0}
    f1 = fb.update(bar1)
    scaled1 = sc.transform(f1[["ret"]])
    sc.update(f1[["ret"]])
    assert scaled1["ret"].iloc[0] == 0.0

    bar2 = {"timestamp": 2, "close": 101.0}
    f2 = fb.update(bar2)
    scaled2 = sc.transform(f2[["ret"]])
    sc.update(f2[["ret"]])
    assert f2["ret"].iloc[0] != 0.0
    assert scaled2["ret"].iloc[0] != 0.0
