import surprise
import numpy as np
import torch


class SurpriseAlgorithm:

    def __init__(self, algo: surprise.prediction_algorithms.AlgoBase, itemdiff: int=0):
        self.algo = algo
        self.itemdiff = itemdiff

    def __create_trainset(self, dataset: torch.Tensor):
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

    def fit(self, trainset):
        trainset = self.__create_trainset(trainset)
        return self.algo.fit(trainset)

    def __call__(self, testset: torch.Tensor):
        i = testset.shape[0]
        testset = testset.numpy()
        # adding empty ratings column
        testset = np.hstack((testset, np.zeros((i, 1))))
        trainset = self.__create_trainset(testset)
        testset = trainset.build_testset()
        predictions = torch.zeros(i, dtype=torch.float64)
        for i, pred in enumerate(self.algo.test(testset)):
            predictions[i] = pred.est
        return predictions