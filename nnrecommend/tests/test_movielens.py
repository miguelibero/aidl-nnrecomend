
import os
import pytest
import numpy as np
from nnrecommend.dataset import BaseDatasetSource
from nnrecommend.dataset.movielens import MovielensLabDatasetSource, Movielens100kDatasetSource
from nnrecommend.hparams import HyperParameters

BASE_PATH = os.path.join(os.path.dirname(__file__), "../../data")
LAB_DATASET_PATH = os.path.join(BASE_PATH, "ml-dataset-splitted/movielens")
HUNDREDK_DATASET_PATH = os.path.join(BASE_PATH, "ml-100ku.data")
SOURCES = (
    MovielensLabDatasetSource(LAB_DATASET_PATH),
    Movielens100kDatasetSource(HUNDREDK_DATASET_PATH),
)

@pytest.mark.parametrize("src", SOURCES)
def test_dataset(src: BaseDatasetSource):
    hparams = HyperParameters({'interaction_context': None})
    src.load(hparams)
    src.trainset.add_negative_sampling(hparams.negatives_train, src.useritems)
    src.testset.add_negative_sampling(hparams.negatives_test, src.useritems)

    assert len(src.trainset) == 5*99057
    assert (src.trainset.idrange == (943, 2625)).all()
    assert src.useritems.shape == (2625, 2625)
    assert len(src.testset) == 943*100
    assert (src.testset.idrange == (943, 2625)).all()

    assert np.equal((0, 113+943, 1), src.trainset).all(axis=1).any()
    assert np.equal((861, 1008+943, 1), src.trainset).all(axis=1).any()
    assert np.equal((500, 1949, 1), src.testset).all(axis=1).any()