import torch.utils.data
import scipy.sparse as sp
import numpy as np
from typing import Container


class Dataset(torch.utils.data.Dataset):
    """
    basic dataset class
    """
    def __init__(self, interactions: np.ndarray):
        """
        :param interactions: 2d array with columns (user id, item id, label, features...)
        """

        interactions = np.array(interactions).astype(int)
        assert len(interactions.shape) == 2 # should be two dimensions
        if interactions.shape[1] == 2:
            # if the interactions don't come with label column, create it with ones
            interactions = np.c_[interactions, np.ones(interactions.shape[0], int)]
        assert interactions.shape[1] > 2 # should have at least 3 columns

        self.__interactions = interactions
        self.idrange = None

    def normalize_ids(self, iddiff: np.ndarray=None) -> np.ndarray:
        """
        if not iddiff parameter is passed, method will calculate one
        so that the dataset has normalized user & item ids to start with 0 and be consecutive

        :param iddiff: two int values that represent the diff for the user and item ids to apply
        """
        idmax = np.max(self.__interactions[:, :2], axis=0)
        if isinstance(iddiff, type(None)):
            idmin = np.min(self.__interactions[:, :2], axis=0)
            iddiff = -idmin
            iddiff[1] += idmax[0] - idmin[0] + 1
        else:
            iddiff = np.array(iddiff).astype(int)
            assert len(iddiff.shape) == 1
            assert iddiff.shape[0] == 2

        self.idrange = idmax + iddiff + 1
        self.__interactions[:, :2] += iddiff
        return iddiff

    def __len__(self) -> int:
        return len(self.__interactions)

    def __getitem__(self, index) -> np.ndarray:
        return self.__interactions[index]

    def __get_random_item(self) -> int:
        """
        return a valid random item id
        """
        assert self.idrange is not None
        return np.random.randint(self.idrange[0], self.idrange[1])

    def get_random_negative_items(self, container: Container, user: int, item: int, num: int=1) -> np.ndarray:
        """
        return an array of random item ids that meet certain conditions
        TODO: this method can produce infinite loops if there is no item that meets the requirements

        :param container: container to check if the interaction exists (usually the adjacency matrix)
        :param user: should not have an interaction with the given user
        :param item: should not be this item
        :param num: length of the array
        """
        items = np.zeros(num, int)
        for i in range(num):
            j = self.__get_random_item()
            while j == item or (user, j) in container:
                j = self.__get_random_item()
            items[i] = j
        return items

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
        return testset


    def create_adjacency_matrix(self) -> sp.dok_matrix:
        """
        create the adjacency matrix for the dataset
        """
        if self.idrange is None:
            self.normalize_ids()
        size = self.idrange[1]
        matrix = sp.dok_matrix((size, size), dtype=np.float32)
        for row in self.__interactions:
            user, item = row[:2]
            matrix[user, item] = 1.0
            matrix[item, user] = 1.0
        return matrix

