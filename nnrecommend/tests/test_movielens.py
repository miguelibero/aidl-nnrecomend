
import os
import numpy as np
from nnrecommend.movielens import MovieLens100kDataset


DATASET_PATH = os.path.join(os.path.dirname(__file__), "../../datasets/ml-dataset-splitted/movielens")


def test_dataset():
    print(DATASET_PATH)
    dataset = MovieLens100kDataset(DATASET_PATH)
    assert len(dataset) == 5*99057
    assert 2624 == dataset.train_mat.shape[0]
    assert len(dataset.test_set) == 943
    assert (dataset[50] == (0, 1122, 1)).all()
    assert (dataset[90000] == (186, 1010, 1)).all()
    assert (dataset.test_set[500][0] == (500, 1948)).all()