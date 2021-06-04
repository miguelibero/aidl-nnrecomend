import surprise
import numpy as np
import torch
from nnrecommend.hparams import HyperParameters
from surprise.prediction_algorithms.algo_base import AlgoBase


class SurpriseAlgorithm:

    def __init__(self, algo: surprise.prediction_algorithms.AlgoBase, itemdiff: int=0):
        self.algo = algo
        self.itemdiff = itemdiff

    def __create_surprise_trainset(self, dataset: torch.Tensor):
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
            u, i, r = row[0:3].tolist()
            u, i = int(u), int(i)
            fi = i - self.itemdiff
            add_rating(ur, u, fi, r)
            add_rating(ir, fi, u, r)
            raw2inner_id_users[u] = u
            raw2inner_id_items[i] = fi

        n_users = len(ur) 
        n_items = len(ir)
        n_ratings = len(dataset)
        ratings = dataset[:, 2]
        rating_scale = (
            np.min(ratings).item(),
            np.max(ratings).item(),
        )
        return surprise.Trainset(ur, ir, n_users, n_items, n_ratings, rating_scale,
                 raw2inner_id_users, raw2inner_id_items)

    def fit(self, trainset: torch.Tensor):
        strainset = self.__create_surprise_trainset(trainset)
        return self.algo.fit(strainset)

    def __call__(self, testset: torch.Tensor):
        i = testset.shape[0]
        testset = testset.numpy()
        # adding empty ratings column
        testset = np.hstack((testset, np.zeros((i, 1))))
        strainset = self.__create_surprise_trainset(testset)
        stestset = strainset.build_testset()
        predictions = torch.zeros(i, dtype=torch.float64)
        for i, pred in enumerate(self.algo.test(stestset)):
            predictions[i] = pred.est
        return predictions



ALGORITHM_TYPES = ['baseline', 'normal', 'slope', 'cocluster', 'knn', 'knn-means', 'svd', 'nmf']

def create_algorithm(algorithm_type, hparams: HyperParameters, idrange: np.ndarray):
    algo = create_surprise_algorithm(algorithm_type, hparams)
    return SurpriseAlgorithm(algo, idrange[0])


def create_surprise_algorithm(algorithm_type, hparams: HyperParameters) -> AlgoBase:
    if algorithm_type == "knn":
        return surprise.prediction_algorithms.knns.KNNBasic()
    if algorithm_type == "knn-means":
        return surprise.prediction_algorithms.knns.KNNWithMeans()
    elif algorithm_type == "svd":
        return surprise.prediction_algorithms.matrix_factorization.SVD()
    elif algorithm_type == "svdpp": # takes long time
        return surprise.prediction_algorithms.matrix_factorization.SVDpp()
    elif algorithm_type == "nmf": # float division error
        return surprise.prediction_algorithms.matrix_factorization.NMF()
    elif algorithm_type == "cocluster":
        return surprise.prediction_algorithms.co_clustering.CoClustering()
    elif algorithm_type == "slope":
        return surprise.prediction_algorithms.slope_one.SlopeOne()
    elif algorithm_type == "normal":
        return surprise.prediction_algorithms.random_pred.NormalPredictor()
    elif algorithm_type == "baseline":
        return surprise.prediction_algorithms.baseline_only.BaselineOnly()
    else:
        raise Exception("could not create algorithm")
