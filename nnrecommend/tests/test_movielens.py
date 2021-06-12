
import os
from nnrecommend.dataset.movielens import MovielensLabDatasetSource
from nnrecommend.hparams import HyperParameters


DATASET_PATH = os.path.join(os.path.dirname(__file__), "../../data/ml-dataset-splitted/movielens")


def test_dataset():
    src = MovielensLabDatasetSource(DATASET_PATH)
    hparams = HyperParameters()
    src.load(hparams)
    src.trainset.add_negative_sampling(src.matrix, hparams.negatives_train)
    src.testset.add_negative_sampling(src.matrix, hparams.negatives_test)

    assert len(src.trainset) == 5*99057
    assert (src.trainset.idrange == (943, 2623)).all()
    assert src.matrix.shape == (2623, 2623)
    assert len(src.testset) == 943*100
    assert (src.testset.idrange == (943, 2623)).all()
    assert (src.trainset[50*5] == (0, 113+943, 1)).all()
    assert (src.trainset[90000*5] == (861, 1008+943, 1)).all()
    assert (src.testset[500*100] == (500, 1949, 1)).all()
