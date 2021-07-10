from nnrecommend.dataset import InteractionDataset
import surprise
import numpy as np
import torch
from nnrecommend.hparams import HyperParameters
from surprise.prediction_algorithms.algo_base import AlgoBase


class SurpriseAlgorithm:

    def __init__(self, algo: surprise.prediction_algorithms.AlgoBase, itemdiff: int=0):
        self.algo = algo
        self.itemdiff = itemdiff

    def __create_surprise_trainset(self, dataset: InteractionDataset):
        ur = {}
        ir = {}
        raw2inner_id_users = {}
        raw2inner_id_items = {}
        def add_rating(cont, one, other, rating):
            if one not in cont:
                subcont = []
                cont[one] = subcont
            else:
                subcont = cont[one]
            subcont.append((other, rating))

        for row in dataset:
            u, i, r = int(row[0]), int(row[1]), row[-1]
            fi = i - self.itemdiff
            add_rating(ur, u, fi, r)
            add_rating(ir, fi, u, r)
            raw2inner_id_users[u] = u
            raw2inner_id_items[i] = fi

        n_users = len(ur) 
        n_items = len(ir)
        n_ratings = len(dataset)
        ratings = dataset[:, -1]
        rating_scale = (
            np.min(ratings).item(),
            np.max(ratings).item(),
        )
        return surprise.Trainset(ur, ir, n_users, n_items, n_ratings, rating_scale,
                 raw2inner_id_users, raw2inner_id_items)

    def fit(self, dataset: InteractionDataset) -> None:
        trainset = self.__create_surprise_trainset(dataset)
        self.algo.fit(trainset)

    def __call__(self, testset: torch.Tensor) -> torch.tensor:
        assert len(testset.shape) == 2
        assert testset.shape[1] > 1
        device = testset.device
        testset = testset.numpy()
        predictions = []
        for row in testset:
            uid = row[0].item()
            iid = row[1].item()
            pred = self.algo.predict(uid, iid)
            predictions.append(pred.est)
        predictions = torch.FloatTensor(predictions)
        if device:
            predictions = predictions.to(device)
        return predictions


ALGORITHM_TYPES = ('baseline', 'normal', 'slope', 'cocluster', 'knn', 'knn-means', 'svd', 'svdpp', 'nmf')
DEFAULT_ALGORITHM_TYPES = ('baseline', 'normal', 'knn')
# seems to match the paper numbers https://arxiv.org/pdf/1909.06627v1.pdf
KNN_DEFAULT_K = 4


def create_algorithm(algorithm_type, hparams: HyperParameters, idrange: np.ndarray):
    algo = create_surprise_algorithm(algorithm_type, hparams)
    return SurpriseAlgorithm(algo, idrange[0])


def create_surprise_algorithm(algorithm_type, hparams: HyperParameters) -> AlgoBase:
    if algorithm_type == "knn":
        return surprise.prediction_algorithms.knns.KNNBasic(verbose=False, k=KNN_DEFAULT_K)
    if algorithm_type == "knn-means":
        return surprise.prediction_algorithms.knns.KNNWithMeans(verbose=False, k=KNN_DEFAULT_K)
    elif algorithm_type == "svd":
        return surprise.prediction_algorithms.matrix_factorization.SVD(verbose=False, lr_all=hparams.learning_rate)
    elif algorithm_type == "svdpp": # takes long time, hangs the computer
        return surprise.prediction_algorithms.matrix_factorization.SVDpp(verbose=False)
    elif algorithm_type == "nmf": # float division error
        return surprise.prediction_algorithms.matrix_factorization.NMF(verbose=False)
    elif algorithm_type == "cocluster":
        return surprise.prediction_algorithms.co_clustering.CoClustering(verbose=False)
    elif algorithm_type == "slope":
        return surprise.prediction_algorithms.slope_one.SlopeOne(verbose=False)
    elif algorithm_type == "normal":
        return surprise.prediction_algorithms.random_pred.NormalPredictor()
    elif algorithm_type == "baseline":
        return surprise.prediction_algorithms.baseline_only.BaselineOnly(verbose=False)
    else:
        raise Exception("could not create algorithm")
