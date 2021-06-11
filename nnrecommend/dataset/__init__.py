import random
import scipy.sparse as sp
import numpy as np
import torch
from logging import Logger
from nnrecommend.logging import get_logger
from typing import Container
from bisect import bisect_left


class Dataset(torch.utils.data.Dataset):
    """
    basic dataset class
    """
    def __init__(self, interactions: np.ndarray, dtype=None):
        """
        :param interactions: 2d array with columns (user id, item id, label, features...)
        """
        interactions = np.array(interactions)
        if dtype:
            interactions = interactions.astype(dtype)
        assert len(interactions.shape) == 2 # should be two dimensions
        if interactions.shape[1] == 2:
            # if the interactions don't come with label column, create it with ones
            interactions = np.c_[interactions, np.ones(interactions.shape[0], interactions.dtype)]
        assert interactions.shape[1] > 2 # should have at least 3 columns

        self.__interactions = interactions
        self.idrange = None

    def denormalize_ids(self, mapping: Container[np.ndarray]):
        assert isinstance(mapping, (list, tuple))
        assert len(mapping) > 1
        mapping = (np.array(mapping[0]), np.array(mapping[1]))

        def find(container, val):
            if val < 0 or val >= len(container):
                return -1
            return container[int(val)]

        lu = len(mapping[0])
        for row in self.__interactions:
            row[0] = find(mapping[0], row[0])
            row[1] = find(mapping[1], row[1] - lu)
        
        self.__remove_negative_ids()

        self.idrange = None


    def normalize_ids(self, mapping: Container[np.ndarray]=None) -> Container[np.ndarray]:
        """
        if mapping parameter is passed, method will calculate one
        so that the dataset has normalized user & item ids to start with 0 and be consecutive
        (item ids start after user ids)

        in case the mapping is passed and some id is not in the list, row will be removed

        :param mapping: an array with two rows of raw user & item ids in order
        """

        def calcmap(ids):
            return np.sort(np.unique(ids))

        if isinstance(mapping, type(None)):
            mapping = (
                calcmap(self.__interactions[:, 0]),
                calcmap(self.__interactions[:, 1]))
        else:
            assert isinstance(mapping, (list, tuple))
            assert len(mapping) > 1
            mapping = (np.array(mapping[0]), np.array(mapping[1]))

        missed = False

        def find(container, val):
            idx = bisect_left(container, val)
            if idx < 0 or idx >= len(container) or container[idx] != val:
                return -1
            return idx

        lu, li = len(mapping[0]), len(mapping[1])
        self.idrange = np.array((lu, lu+li))
        for row in self.__interactions:
            row[0] = find(mapping[0], row[0])
            row[1] = find(mapping[1], row[1]) + lu
        
        self.__remove_negative_ids()

        return mapping

    def __remove_negative_ids(self):
        cond = (self.__interactions[:, :2] >= 0).all(axis=1)
        self.__interactions =  self.__interactions[cond]

    def __len__(self) -> int:
        return len(self.__interactions)

    def __getitem__(self, index) -> np.ndarray:
        return self.__interactions[index]

    def __get_random_item(self) -> int:
        """
        return a valid random item id
        """
        assert self.idrange is not None
        return np.random.randint(self.idrange[0], self.idrange[1], dtype=np.int64)

    def get_random_negative_items(self, container: Container, user: int, item: int, num: int=1) -> np.ndarray:
        """
        return an array of random item ids that meet certain conditions
        TODO: this method can produce infinite loops if there is no item that meets the requirements

        :param container: container to check if the interaction exists (usually the adjacency matrix)
        :param user: should not have an interaction with the given user
        :param item: should not be this item
        :param num: length of the array
        """
        items = set()
        for i in range(num):
            j = self.__get_random_item()
            while j == item or j in items or (user, j) in container:
                j = self.__get_random_item()
            items.add(j)
        return np.array(list(items), dtype=np.int64)

    def get_random_negative_items_2(self, container: Container, user: int, item: int, num: int=1) -> np.ndarray:
        # slower implementation avoiding duplicate items
        assert self.idrange is not None

        items = range(self.idrange[0], self.idrange[1])
        items = [i for i in items if i != item and (user, i) not in container]
        num = min(len(items), num)
        items = random.sample(items, num)
        return np.array(items, dtype=np.int64)

    def add_negative_sampling(self, container: Container, num: int=1) -> None:
        """
        add negative samples to the dataset interactions
        with random item ids that don't match existing interactions
        the negative samples will be placed in the rows immediately after the original one

        :param container: container to check if the interaction exists (usually the adjacency matrix)
        :param num: amount of samples per interaction
        """
        if num <= 0:
            return

        n = num+1
        data = np.repeat(self.__interactions, n, axis=0)
        for i, row in enumerate(self.__interactions):
            data[1+n*i:n*(i+1), 1] = self.get_random_negative_items(container, row[0], row[1], num)
            data[1+n*i:n*(i+1), 2] = 0
        self.__interactions = data

    def extract_test_dataset(self, num_user_interactions: int=1, min_keep_user_interactions: int=1) -> 'Dataset':
        """
        extract the last positive interaction of every user for the test dataset,
        check that the user has a minimum amount of interactions before extracting the last one

        :param num_user_interactions: amount of user interactions to extract to the test dataset
        :param min_keep_user_interactions: minimum amount of user interactions to keep in the original dataset
        """
        if self.idrange is None:
            self.normalize_ids()
        rowsbyuser = {}
        for i, row in enumerate(self.__interactions):
            if row[2] <= 0:
                continue
            u = row[0]
            if u not in rowsbyuser:
                userrows = []
                rowsbyuser[u] = userrows
            else:
                userrows = rowsbyuser[u]
            userrows.append(i)
        rows = []
        for userrows in rowsbyuser.values():
            if len(userrows) < num_user_interactions + min_keep_user_interactions:
                continue
            rows += userrows[-num_user_interactions:]
        testset = Dataset(self.__interactions[rows])
        self.__interactions = np.delete(self.__interactions, rows, axis=0)
        testset.idrange = self.idrange
        return testset


    def create_adjacency_matrix(self) -> sp.spmatrix:
        """
        create the adjacency matrix for the dataset
        """
        if self.idrange is None:
            self.normalize_ids()
        size = self.idrange[1]
        matrix = sp.dok_matrix((size, size), dtype=np.int64)
        for row in self.__interactions:
            user, item = row[:2]
            matrix[user, item] = 1.0
            matrix[item, user] = 1.0
        return matrix

    def __remove_low(self, matrix: sp.spmatrix, lim: int, idx: int) -> None:
        if self.idrange is None:
            self.normalize_ids()
        counts = np.asarray(matrix.sum(0)).flatten()
        ids = self.__interactions[:, idx].astype(np.int64)
        cond = counts[ids] > lim
        self.__interactions =  self.__interactions[cond]
        return np.count_nonzero(cond == False)

    def remove_low(self, matrix: sp.spmatrix, lim: int) -> None:
        return self.remove_low_users(matrix, lim) + self.remove_low_items(matrix, lim)

    def remove_low_users(self, matrix: sp.spmatrix, lim: int) -> None:
        return self.__remove_low(matrix, lim, 0)

    def remove_low_items(self, matrix: sp.spmatrix, lim: int) -> None:
        return self.__remove_low(matrix, lim, 1)


class BaseDatasetSource:

    def __init__(self, logger: Logger=None):
        self._logger = logger or get_logger(self)
        self.trainset = None
        self.testset = None
        self.matrix = None

    def load(self, max_interactions: int=-1):
        raise NotImplementedError()
