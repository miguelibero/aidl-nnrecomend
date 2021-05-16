
import os
from nnrecommend.dataset.movielens import MovielensDataset

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../../datasets/ml-dataset-splitted/movielens")


def test_dataset():
    data = MovielensDataset(DATASET_PATH)
    data.setup(4, 99)

    assert len(data.trainset) == 5*99057
    assert (data.trainset.idrange == (943, 2625)).all()
    assert data.matrix.shape == (2625, 2625)
    assert len(data.testset) == 943*100
    assert (data.testset.idrange == (943, 2595)).all()
    assert (data.trainset[50*5] == (0, 113+943, 1, 875072173)).all()
    assert (data.trainset[90000*5] == (861, 1008+943, 1, 879303622)).all()
    assert (data.testset[500*100] == (500, 1949, 1, 883995203)).all()
