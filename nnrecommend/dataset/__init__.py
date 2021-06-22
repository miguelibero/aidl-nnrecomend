import itertools
import torch
import scipy.sparse as sp
import numpy as np
from logging import Logger
from typing import Any, Container, List, Tuple
from bisect import bisect_left
from nnrecommend.hparams import HyperParameters
from nnrecommend.logging import get_logger


class InteractionDataset(torch.utils.data.Dataset):
    """
    basic interaction dataset class
    """
    def __init__(self, interactions: np.ndarray):
        """
        :param interactions: 2d array with columns (user id, item id, context, label...)
        """
        interactions = np.array(interactions)
        assert len(interactions.shape) == 2 # should be two dimensions
        if interactions.shape[1] == 2:
            # if the interactions don't come with label column, create it with ones
            interactions = np.c_[interactions, np.ones(interactions.shape[0], interactions.dtype)]
        assert interactions.shape[1] > 2 # should have at least 3 columns
        self.__interactions = interactions
        self.idrange = None

    def __validate_mapping(self, mapping: Container[np.ndarray]):
        mapping = [np.array(v) for v in mapping]
        assert isinstance(mapping, (list, tuple))
        assert len(mapping) < self.__interactions.shape[1]
        return mapping

    def denormalize_ids(self, mapping: Container[np.ndarray], remove_missing=True) -> Container[np.ndarray]:
        """
        convert from normalized ids back to the original ones

        :param mapping: a container with each element a numpy array of raw ids in order
        :param remove_missing: remove rows if some of the ids are not in the mapping
        """
        mapping = self.__validate_mapping(mapping)
        i = 0
        diff = 0
        for colmapping in mapping:
            colgen = IdFinder(colmapping)
            for row in self.__interactions:
                row[i] = colgen.reverse(row[i] - diff)
            diff += len(colmapping)
            i += 1
        if remove_missing:
            self.__remove_negative_ids()
        self.idrange = None
        return mapping

    def normalize_ids(self, assume_consecutive=False) -> Container[np.ndarray]:
        """
        calculate a mapping for all the columns except the last one (label)
        so that the values are consecutive integers starting with 0

        :param assume_consecutive: assume the ids are consecutive already (only shift)
        :return mapping: a container with each element a numpy array of raw ids in order
        """
        mapping = []
        for i in range(self.__interactions.shape[1] - 1):
            ids = self.__interactions[:, i]
            if assume_consecutive:
                colmapping = np.arange(np.min(ids), np.max(ids)+1)
            else:
                colmapping = np.sort(np.unique(ids))
            mapping.append(colmapping)
        self.__normalize_ids(mapping, False)
        return mapping

    def map_ids(self, mapping: Container[np.ndarray], remove_missing=True) -> Container[np.ndarray]:
        """
        apply an existing mapping to the ids

        :param mapping: a container with each element a numpy array of raw ids in order
        :param remove_missing: remove rows if some of the ids are not in the mapping
        """
        assert self.idrange is None
        mapping = self.__validate_mapping(mapping)
        self.__normalize_ids(mapping, remove_missing)
        return mapping

    def __normalize_ids(self, mapping: Container[np.ndarray], remove_missing: bool) -> None:
        i = 0
        diff = 0
        self.idrange = np.zeros(len(mapping), dtype=np.int64)
        for colmapping in mapping:
            colgen = IdFinder(colmapping)
            for row in self.__interactions:
                row[i] = colgen.find(row[i]) + diff
            diff += len(colmapping)
            self.idrange[i] = diff
            i += 1
        
        if remove_missing:
            self.__remove_negative_ids()

    def __remove_negative_ids(self):
        cond = (self.__interactions[:, :2] >= 0).all(axis=1)
        self.__interactions =  self.__interactions[cond]

    def __len__(self) -> int:
        return len(self.__interactions)

    def __getitem__(self, index) -> np.ndarray:
        return self.__interactions[index]

    def __get_row_pairs(self, row: np.ndarray) -> Container[np.ndarray]:
        max = -1 if self.idrange is None else len(self.idrange)
        return itertools.combinations(row[:max], 2)

    MAX_RANDOM_TRIES = 1000

    def get_random_negative_row(self, row: np.ndarray, container: Container = None) -> np.ndarray:
        """
        generates a random negative row, by default the context values
        will be the same as the positive row, use rand_context=True to get random values
        for them too
        :param row: the positive row
        :param container: pass a container to check that the value is not in it
        """
        row = np.array(row)
        assert self.idrange is not None
        assert row.shape[0] >= len(self.idrange)
        c = 0
        v = None
        minv, maxv = self.idrange[0:2]
        while True:
            if c > self.MAX_RANDOM_TRIES:
                return None
            v = np.random.randint(minv, maxv, dtype=np.int64)
            c += 1
            if v == row[1]:
                continue
            if container is not None and tuple(row[0], v) in container:
                continue
            break
        nrow = row.copy()
        nrow[-1] = 0
        nrow[1] = v
        return nrow

    def get_random_negative_rows(self, container: Container, row: np.ndarray, num: int=1) -> np.ndarray:
        """
        generate num random rows that don't have values in row and are not in the container

        current implementation may throw after some time if it's not possible to find empty pairs in the container
        """
        nrows = []
        for i in range(num):
            nrow = self.get_random_negative_row(row)
            if nrow is not None:
                nrows.append(nrow)
        return np.vstack(nrows)

    def add_negative_sampling(self, container: Container, num: int=1) -> None:
        """
        add negative samples to the dataset interactions
        with random ids that don't match existing interactions
        the negative samples will be placed in the rows immediately after the original one

        :param container: container to check if the interaction exists (usually the adjacency matrix)
        :param num: amount of samples per interaction
        """
        assert num > 0
        n = num + 1
        data = np.repeat(self.__interactions, n, axis=0)
        for i, row in enumerate(self.__interactions):
            data[1+n*i:n*(i+1), :] = self.get_random_negative_rows(container, row, num)
        self.__interactions = data

    def __require_normalized(self):
        if self.idrange is None:
            self.normalize_ids()

    def extract_negative_dataset(self) -> None:
        """
        extract a new dataset with the negative values
        """
        cond = self.__interactions[:, -1] == 0
        negset = InteractionDataset(self.__interactions[cond])
        self.__interactions = np.delete(self.__interactions, cond, axis=0)
        negset.idrange = self.idrange
        return negset

    def extract_test_dataset(self, num_user_interactions: int=1, min_keep_user_interactions: int=1, take_bottom:bool=False) -> 'InteractionDataset':
        """
        extract the first or last positive interaction of every user for the test dataset,
        check that the user has a minimum amount of interactions before extracting the last one

        :param num_user_interactions: amount of user interactions to extract to the test dataset
        :param min_keep_user_interactions: minimum amount of user interactions to keep in the original dataset
        :param take_bottom: set to true to take the last interactions
        :return: test Dataset
        """
        self.__require_normalized()
        rowsbyuser = {}
        for i, row in enumerate(self.__interactions):
            if row[-1] <= 0:
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
            if take_bottom:
                rows += userrows[-num_user_interactions:]
            else:
                rows += userrows[:num_user_interactions]

        testset = InteractionDataset(self.__interactions[rows])
        self.__interactions = np.delete(self.__interactions, rows, axis=0)
        testset.idrange = self.idrange
        return testset

    def create_adjacency_matrix(self) -> sp.spmatrix:
        """
        create the adjacency matrix for the dataset
        """
        self.__require_normalized()
        size = self.idrange[-1]
        matrix = sp.dok_matrix((size, size), dtype=np.int64)
        for row in self.__interactions:
            for a, b in self.__get_row_pairs(row):
                matrix[a, b] = 1
                matrix[b, a] = 1
        return matrix

    def __get_col_range(self, col: int) -> None:
        self.__require_normalized()
        assert col >= 0 and col < len(self.idrange)
        start = self.idrange[col-1] if col > 0 else 0
        end = self.idrange[col]
        return start, end

    def __get_submatrix(self, matrix: sp.spmatrix, col1: int, col2: int):
        rs, re = self.__get_col_range(col1)
        cs, ce = self.__get_col_range(col2)
        return matrix[rs:re, cs:ce]

    def remove_low(self, matrix: sp.spmatrix, lim: int, col1: int, col2: int) -> int:
        self.__require_normalized()
        submatrix = self.__get_submatrix(matrix, col1, col2)
        counts = np.asarray(submatrix.sum(1)).flatten()
        ids = self.__interactions[:, col1].astype(np.int64)
        ids -= self.idrange[col1]
        cond = counts[ids] > lim
        self.__interactions =  self.__interactions[cond]
        return np.count_nonzero(cond == False)

    def remove_low_users(self, matrix: sp.spmatrix, lim: int) -> int:
        return self.remove_low(matrix, lim, 0, 1)

    def remove_low_items(self, matrix: sp.spmatrix, lim: int) -> int:
        return self.remove_low(matrix, lim, 1, 0)

    def remove_low_all(self, matrix: sp.spmatrix, lim: int) -> int:
        self.__require_normalized()
        count = 0
        cols = len(self.idrange)
        for (col1, col2) in itertools.combinations(range(0, cols), 2):
            count += self.remove_low(matrix, lim, col1, col2)
        return count
    
    def add_previous_item_column(self):
        """
        adds a new context column with the values of the previous item
        by the same user. The values are consecutive to the last column
        range and the first value represents no previous item.
        """
        self.__require_normalized()

        # fill the new column with zeros
        col = np.zeros(self.__interactions.shape[0])
        self.__interactions = np.insert(self.__interactions, -1, col, axis=1)

        for i in range(self.idrange[0]):
            # find all the interactions of a user
            cond = self.__interactions[:, 0] == i
            # get the items
            items = self.__interactions[cond, 1]
            # shift them so they start with 0
            items -= self.idrange[0]
            # add a -1 in the beginning and remove the last one
            items = np.insert(items, 0, -1)[:-1]
            # shift them to the end of the last range
            items += self.idrange[-1] + 1
            # assign them to the new column
            self.__interactions[cond, -2] = items

        r = np.max(self.__interactions[:, -2]) if self.__interactions.shape[0] > 0 else 0
        self.idrange = np.append(self.idrange, r + 1)


class InteractionPairDataset(torch.utils.data.Dataset):
    """
    returns pairs of positive and negative interactions
    """

    def __init__(self, positive: torch.utils.data.Dataset, negative: torch.utils.data.Dataset):
        self.positive = positive
        self.negative = negative

    def __len__(self) -> int:
        return max(len(self.positive), len(self.negative))

    def __getitem__(self, index) -> Tuple:
        f = len(self.negative) / len(self.positive)
        pos = int(index / f)
        neg = index
        if f < 1:
            pos, neg = neg, pos
        return (self.positive[pos], self.negative[neg])


class BaseDatasetSource:
    """
    basic class to load a dataset
    both trainset and testset should have the following structure
    (user id, item id, rating, context 1, context 2)

    user id, item id and context columns should be normalized,
    meaning that the values are consecutive and don't overlap

    item_info should be a dictionary with item_id keys
    and values should be a dictionary with the diferent decriptive fields
    """

    def __init__(self, logger: Logger=None):
        self._logger = logger or get_logger(self)
        self.trainset = None
        self.testset = None
        self.matrix = None
        self.item_info = None

    def load(self, hparams: HyperParameters):
        raise NotImplementedError()


def save_model(path: str, model, src: BaseDatasetSource):
    data = {
        "model": model,
        "idrange": src.trainset.idrange,
        "item_info": src.item_info
    }
    with open(path, "wb") as fh:
        torch.save(data, fh)


class IdFinder:

    def __init__(self, data: Container=[], hash: bool=False):
        self.data = data
        self.__hash = hash

    def _fix(self, v):
        if self.__hash:
            v = hash(v)
        return v

    def find(self, v) -> int:
        v = self._fix(v)
        id = bisect_left(self.data, v)
        if not self._check(id, v):
            return -1
        return id

    def _check(self, id, v):
        return id >= 0 and id < len(self.data) and self.data[id] == v

    def reverse(self, v: int) -> Any:
        v = int(v)
        if v < 0 or v >= len(self.data):
            return -1
        return self.data[v]


class IdGenerator(IdFinder):

    def __init__(self, hash: bool=False):
        super().__init__([], hash)

    def add(self, v) -> None:
        v = self._fix(v)
        id = bisect_left(self.data, v)
        if not self._check(id, v):
            self.data.insert(id, v)
