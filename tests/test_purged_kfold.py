import numpy as np
from quant_pipeline.training import PurgedKFold


def test_purged_kfold_embargo():
    pkf = PurgedKFold(n_splits=3, embargo=1)
    X = np.arange(9)
    splits = list(pkf.split(X))
    assert len(splits) == 3

    train0, test0 = splits[0]
    assert np.array_equal(test0, np.array([0, 1, 2]))
    assert np.array_equal(train0, np.array([4, 5, 6, 7, 8]))

    train1, test1 = splits[1]
    assert np.array_equal(test1, np.array([3, 4, 5]))
    assert np.array_equal(train1, np.array([0, 1, 7, 8]))

    train2, test2 = splits[2]
    assert np.array_equal(test2, np.array([6, 7, 8]))
    assert np.array_equal(train2, np.array([0, 1, 2, 3, 4]))
