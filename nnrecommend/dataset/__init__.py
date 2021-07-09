import torch
import random
import scipy.sparse as sp
import numpy as np
import itertools
from pandas.core.frame import DataFrame
from logging import Logger
from bisect import bisect_left
from typing import Any, Container, Dict, Tuple
from nnrecommend.hparams import HyperParameters
from nnrecommend.logging import get_logger


class InteractionDataset(torch.utils.data.Dataset):
    """
    basic interaction dataset class
    """
    def __init__(self, interactions: np.ndarray, add_labels_col: bool=False):
        """
        :param interactions: 2d array with columns (user id, item id, context, label...)
        """
        interactions = np.array(interactions)
        assert len(interactions.shape) == 2 # should be two dimensions
        assert interactions.shape[1] > 1 # should have at least 2 columns
        if add_labels_col or interactions.shape[1] == 2:
            # if the interactions don't come with label column, create it with ones
            interactions = np.c_[interactions, np.ones(interactions.shape[0], interactions.dtype)]
        self.__interactions = interactions
        self.idrange = None

    def __validate_mapping(self, mapping: Container[np.ndarray]) -> Container[np.ndarray]:
        mapping = [np.array(v) for v in mapping]
        assert isinstance(mapping, (list, tuple))
        assert len(mapping) < self.__interactions.shape[1]
        return mapping

    def denormalize_ids(self, mapping: Container[np.ndarray]) -> Container[np.ndarray]:
        """
        convert from normalized ids back to the original ones

        :param mapping: a container with each element a numpy array of raw ids in order
        """
        mapping = self.__validate_mapping(mapping)
        i = 0
        diff = 0
        missing = set()
        for colmapping in mapping:
            colgen = IdFinder(colmapping)
            for j, row in enumerate(self.__interactions):
                v = colgen.reverse(row[i] - diff)
                if v is None:
                    missing.add(j)
                else:
                    row[i] = v
            diff += len(colmapping)
            i += 1
        self.__interactions = np.delete(self.__interactions, list(missing), 0)
        self.idrange = None
        return mapping

    def __require_normalized(self) -> None:
        if self.idrange is None:
            self.normalize_ids()

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
            colmapping = self.__get_col_mapping(ids, assume_consecutive)
            mapping.append(colmapping)
        self.__normalize_ids(mapping)
        return mapping

    def __get_col_mapping(self, values: np.ndarray, assume_consecutive=False) -> np.ndarray:
        if assume_consecutive:
            return np.arange(np.min(values), np.max(values)+1)
        return np.sort(np.unique(values))

    def map_ids(self, mapping: Container[np.ndarray]) -> Container[np.ndarray]:
        """
        apply an existing mapping to the ids

        :param mapping: a container with each element a numpy array of raw ids in order
        """
        assert self.idrange is None
        mapping = self.__validate_mapping(mapping)
        self.__normalize_ids(mapping)
        return mapping

    def __normalize_ids(self, mapping: Container[np.ndarray]) -> None:
        diff = 0
        self.idrange = np.zeros(len(mapping), dtype=np.int64)
        missing = set()
        for i, colmapping in enumerate(mapping):
            ids = self.__interactions[:, i]
            ids, colmissing = self.__normalize_col(ids, colmapping, diff)
            self.__interactions[:, i] = ids
            missing.update(colmissing)
            diff += len(colmapping)
            self.idrange[i] = diff
        self.__interactions = np.delete(self.__interactions, list(missing), 0)

    def __normalize_col(self, values: np.ndarray, colmapping: np.ndarray, minv: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        :returns: normalized items, rows with missing items
        """
        missing = []
        colgen = IdFinder(colmapping)
        for i, v in enumerate(values):
            v = colgen.find(v)
            if v is None:
                missing.append(i)
            else:
                values[i] = v + minv
        return values, missing

    def get_grounded(self) -> np.ndarray:
        """
        :returns: the interactions with id columns starting with zero
        """
        self.__require_normalized()
        interactions = self.__interactions.copy()
        for i, maxv in enumerate(self.idrange[:-1]):
            interactions[:, i+1] -= maxv
        return interactions

    def __len__(self) -> int:
        return len(self.__interactions)

    def __getitem__(self, index) -> np.ndarray:
        return self.__interactions[index]

    def __setitem__(self, index, val: np.ndarray) -> None:
        self.__interactions[index] = val

    def __get_row_pairs(self, row: np.ndarray) -> Container[np.ndarray]:
        max = -1 if self.idrange is None else len(self.idrange)
        return itertools.combinations(row[:max], 2)

    MAX_RANDOM_TRIES = 1000

    def get_random_negative_item(self, user: int, item: int, container: Container = None) -> int:
        """
        generate a random negative item, will keep trying to generate an item that is not in
        the container or the item parameter value
        :param user: the user of the positive interaction
        :param user: the item of the positive interaction
        :param container: pass a container to check that the interaction is not in it
        """
        assert self.idrange is not None
        c = 0
        v = None
        minv, maxv = self.idrange[0:2]
        while True:
            if c > self.MAX_RANDOM_TRIES:
                raise ValueError("too many random tries")
            v = np.random.randint(minv, maxv, dtype=np.int64)
            c += 1
            if v == item:
                continue
            if container is not None and (user, v) in container:
                continue
            break
        return v

    def get_random_negative_items(self, user: int, item: int, num: int=1, container: Container=None) -> np.ndarray:
        """
        generate a list of random negative items, much faster than the unique method but
        may produce duplicate negative items (usually ok for the trainset)

        :param user: the user id of the positive interaction
        :param item: the item id of the positive interaction
        :param num: amount of items to generate
        :param container: container to check if the interaction exists (usually the adjacency matrix)
        """
        items = []
        for i in range(num):
            items.append(self.get_random_negative_item(user, item, container))
        return np.array(items)

    def get_unique_random_negative_items(self, user: int, item: int, num: int=None, container: Container=None) -> np.ndarray:
        """
        generate a list of random negative items without repeats, this is slower but more suited
        for the testset to guarantee the same amount of negative items when evaluating the performance

        :param user: the user id of the positive interaction
        :param item: the item id of the positive interaction
        :param num: amount of items to generate (negative or none means all the possible candidates)
        :param container: container to check if the interaction exists (usually the adjacency matrix)
        """
        assert self.idrange is not None
        items = [i for i in range(self.idrange[0], self.idrange[1]) if i != item]
        if container is not None:
            items = [i for i in items if (user, i) not in container]
        if isinstance(num, int) and num >= 0:
            if len(items) < num:
                raise ValueError("not enough items to generate random negatives")
            items = random.sample(items, num)
        return np.array(items)

    def get_random_negative_rows(self, row: np.ndarray, num: int=None, container: Container=None, unique: bool=False) -> np.ndarray:
        """
        generate num random rows that don't have the item in row and are not in the container

        :param row: the positive row
        :param num: amount of rows to generate (None or negative means all possible items)
        :param container: container to check if the interaction exists (usually the adjacency matrix)
        :param unique: if the items for each user should not be repeated (slower)
        """
        row = np.array(row)
        assert len(row.shape) == 1
        assert row.shape[0] > 1
        if not isinstance(num, int) or num < 0:
            unique = True
        func = self.get_unique_random_negative_items if unique else self.get_random_negative_items
        items = func(row[0], row[1], num, container)
        if not isinstance(items, np.ndarray) or len(items.shape) != 1:
            raise ValueError("could not generate enough random items")
        nrows = np.repeat(np.expand_dims(row, 0), len(items), axis=0)
        nrows[:, 1] = items
        nrows[:, -1] = 0
        return nrows

    def add_negative_sampling(self, num: int=1, container: Container=None, unique: bool=False) -> Container[np.ndarray]:
        """
        add negative samples to the dataset interactions
        with random item ids that don't match existing interactions
        the negative samples will be placed at the end of the interactions array
        we're trying to optimize memory consumption peaks by resizing the interactions array

        real negative samples will be added to sampling groups of the same user

        :param num: amount of negative samples per interaction (None or negative means add all possible)
        :param container: container to check if the interaction exists (usually the adjacency matrix)
        :param unique: if the items for each user should not be repeated (slower)
        :return: container of arrays with row indices for every group
        """
        self.__require_normalized()
        indices = []
        p = len(self.__interactions)
        c = self.__interactions.shape[1]
        neg_byuser = {}
        indices_byuser = {}

        def add_byuser(container: Dict[int, Container[int]], u: int, v: int) -> Container[int]:
            if u in container:
                group = container[u]
            else:
                group = []
                container[u] = group
            group.append(v)
            return group

        for i in range(p):
            row = self.__interactions[i]
            u = row[0]
            if not self.__is_row_positive(row):
                add_byuser(neg_byuser, u, i)
                continue
            nrows = self.get_random_negative_rows(row, num, container, unique)
            del row
            n = p + len(nrows)
            self.__interactions.resize((n, c), refcheck=False)
            self.__interactions[p:] = nrows
            del nrows
            add_byuser(indices_byuser, u, len(indices))
            gidx = np.arange(p, n, dtype=np.int64)
            gidx = np.insert(gidx, 0, i)
            indices.append(gidx)
            p = n
        # adding real negative interactions to the end of the groups of the same user
        for u, rows in neg_byuser.items():
            if u not in indices_byuser:
                continue
            group = indices_byuser[u]
            for i in rows:
                j = group[i % len(group)]
                indices[j] = np.append(indices[j], i)

        return indices

    def __is_row_positive(self, row: np.ndarray) -> bool:
        """
        check if row is considered a positive interaction
        """
        return row[-1] > 0

    def extract_negative_dataset(self) -> 'InteractionDataset':
        """
        extract a new dataset with the negative values
        """
        cond = self.__interactions[:, -1] == 0
        negset = InteractionDataset(self.__interactions[cond])
        old = self.__interactions
        self.__interactions = self.__interactions[~cond]
        del old
        negset.idrange = self.idrange.copy()
        return negset

    def extract_test_dataset(self, num_user_interactions: int=1, min_keep_user_interactions: int=1, take_bottom:bool=True) -> 'InteractionDataset':
        """
        extract positive interactions of every user for the test dataset,
        check that the user has a minimum amount of interactions before extracting

        :param num_user_interactions: amount of user interactions to extract to the test dataset
        :param min_keep_user_interactions: minimum amount of user interactions to keep in the original dataset
        :param take_bottom: set to true to take the last interactions
        :return: test Dataset
        """
        self.__require_normalized()
        rowsbyuser = {}
        for i, row in enumerate(self.__interactions):
            if not self.__is_row_positive(row):
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
            del userrows

        testset = InteractionDataset(self.__interactions[rows])
        self.__interactions = np.delete(self.__interactions, rows, axis=0)
        testset.idrange = self.idrange.copy()
        return testset

    def create_adjacency_submatrix(self, col1: int = 0, col2: int = 1) -> sp.spmatrix:
        """
        create the adjacency submatrix for the dataset
        """
        self.__require_normalized()
        min1, max1 = self.__get_col_range(col1)
        min2, max2 = self.__get_col_range(col2)
        size = max1 - min1 + max2 - min2
        matrix = sp.dok_matrix((size, size), dtype=np.int64)
        for row in self.__interactions:
            a = row[col1] - min1
            b = row[col2] - min2 + max1
            matrix[a, b] = 1
            matrix[b, a] = 1
        return matrix

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

    def __normalize_col_num(self, col: int):
        return col % self.__interactions.shape[1]

    def __get_col_range(self, col: int) -> Tuple[int]:
        self.__require_normalized()
        col = self.__normalize_col_num(col)
        assert col >= 0 and col < len(self.idrange)
        start = self.idrange[col-1] if col > 0 else 0
        end = self.idrange[col]
        return start, end

    def __get_submatrix(self, matrix: sp.spmatrix, col1: int, col2: int) -> sp.spmatrix:
        rs, re = self.__get_col_range(col1)
        cs, ce = self.__get_col_range(col2)
        return matrix[rs:re, cs:ce]

    def remove_random(self, size: int):
        """
        remove random rows until the dataset has size
        """
        assert size >= 0
        v = len(self.__interactions)
        if size > v:
            return
        rows = np.random.choice(np.arange(0, v), size)
        self.__interactions = self.__interactions[rows]

    def remove_low(self, matrix: sp.spmatrix, lim: int, col1: int, col2: int) -> int:
        """
        remove rows that have under a given amount of duplicated pairs
        :param lim: remove rows with less interactions than this value
        :param col1: number of the first column (used as group)
        :param col2: number of the second column (that will be counted)
        :returns: amount of removed rows
        """
        assert lim >= 0
        if lim == 0:
            return 0
        self.__require_normalized()
        submatrix = self.__get_submatrix(matrix, col1, col2)
        counts = np.asarray(submatrix.sum(1)).flatten()
        minv = 0 if col1 == 0 else self.idrange[col1 - 1]
        ids = self.__interactions[:, col1] - minv
        cond = counts[ids] > lim
        self.__interactions =  self.__interactions[cond]
        return np.count_nonzero(cond == False)

    def remove_low_users(self, matrix: sp.spmatrix, lim: int) -> int:
        """
        remove rows that users with low amount of items
        """
        return self.remove_low(matrix, lim, 0, 1)

    def remove_low_items(self, matrix: sp.spmatrix, lim: int) -> int:
        """
        remove rows that items with low amount of users
        """
        return self.remove_low(matrix, lim, 1, 0)

    def keep_top(self, matrix: sp.spmatrix, amount: int, col1: int, col2: int) -> int:
        """
        keep only the interactions with highest amount of counts of col2 in col1
        """
        assert amount >= 0
        self.__require_normalized()
        submatrix = self.__get_submatrix(matrix, col1, col2)
        if amount >= submatrix.shape[0]:
            return 0
        counts = np.asarray(submatrix.sum(1)).flatten()
        topids = counts.argsort()[-amount:]
        minv = 0 if col1 == 0 else self.idrange[col1 - 1]
        ids = self.__interactions[:, col1] - minv
        cond = np.isin(ids, topids)
        self.__interactions =  self.__interactions[cond]
        return np.count_nonzero(cond == False)

    def keep_top_users(self, matrix: sp.spmatrix, amount: int) -> int:
        """
        remove rows that users with low amount of items
        """
        return self.keep_top(matrix, amount, 0, 1)

    def keep_top_items(self, matrix: sp.spmatrix, amount: int) -> int:
        """
        remove rows that items with low amount of users
        """
        return self.keep_top(matrix, amount, 1, 0)

    def remove_low_all(self, matrix: sp.spmatrix, lim: int) -> int:
        self.__require_normalized()
        count = 0
        cols = len(self.idrange)
        for (col1, col2) in itertools.combinations(range(0, cols), 2):
            count += self.remove_low(matrix, lim, col1, col2)
        return count

    def add_random_column(self, amount: int, insert_col: int=-1):
        """
        :param amount: the amount of items to add
        :param insert_col: the column where to insert
        """
        colmapping = np.arange(0, amount)
        h = self.__interactions.shape[0]
        values = np.random.randint(0, amount, (h, ), dtype=np.int64)
        self.insert_column(insert_col, values, colmapping)
    
    def add_previous_item_column(self, items_col: int=1, insert_col: int=-1) -> None:
        """
        adds a new context column with the values of the previous item
        by the same user. The values are consecutive to the last column
        range and the first value represents no previous item.

        the interaction rows need to be already sorted from older to newer.

        :param items_col: the column index where the items are
        :param insert_col: the column index where to insert
        """
        self.__require_normalized()

        minv, maxv = self.__get_col_range(items_col)

        values = np.zeros(self.__interactions.shape[0])
        for i in range(self.idrange[0]):
            # find all the interactions of a user
            cond = self.__interactions[:, 0] == i
            # get the items
            items = self.__interactions[cond, items_col]
            # shift them so they start with 0
            items -= minv
            # add a -1 in the beginning and remove the last one
            items = np.insert(items, 0, -1)[:-1]
            # add user items to the values
            values[cond] = items

        colmapping = np.arange(-1, maxv - minv)
        self.insert_column(insert_col, values, colmapping)

    def insert_column(self, col: int, values: np.ndarray, colmapping: np.ndarray=None) -> np.ndarray:
        """
        inserts a column into the dataset, adjusting ranges
        :returns: mapping for the new column
        """
        self.__require_normalized()
        col = self.__normalize_col_num(col)
        minv = 0 if col == 0 else self.idrange[col-1]
        if colmapping is None:
            colmapping = self.__get_col_mapping(values)
        values, missing = self.__normalize_col(values, colmapping, minv)
        self.__interactions = np.delete(self.__interactions, list(missing), 0)
        diff = len(colmapping)
        for i in range(col+1, len(self.idrange)):
            self.__interactions[:, i] += diff
            self.idrange[i] += diff
        self.__interactions = np.insert(self.__interactions, col, values, axis=1)
        self.idrange = np.insert(self.idrange, col, minv + diff)
        return colmapping

    def combine_columns(self, base_col: int, *other_cols: Container[int]) -> None:
        """
        combines multiple columns into one, updating idanges too
        combining means that the values will be multiplied so 
        we end up with a normalized column

        :param base_col: column number that will be replaced with the combination
        :param other_cols: column numbers to combine
        """
        self.__require_normalized()
        base_col = self.__normalize_col_num(base_col)
        minb, maxb = self.__get_col_range(base_col)
        brange = maxb - minb

        for col in sorted(other_cols, reverse=True):
            col = self.__normalize_col_num(col)
            minv, maxv = self.__get_col_range(col)
            vals = self.__interactions[:, col] - minv
            range = maxv - minv
            self.__interactions[:, base_col] += vals*brange
            brange *= range
            self.__interactions = np.delete(self.__interactions, col, 1)
            self.idrange = np.delete(self.idrange, col)
        self.idrange[base_col] = minb + brange

    def remove_column(self, col: int) -> None:
        """
        remove dataset column, adjust ranges
        """
        self.__require_normalized()
        col = self.__normalize_col_num(col)
        minv, maxv = self.__get_col_range(col)
        diff = maxv - minv
        for i in range(col + 1, len(self.idrange)):
            self.idrange[i] -= diff
            self.__interactions[:, i] -= diff
        self.__interactions = np.delete(self.__interactions, col, 1)
        self.idrange = np.delete(self.idrange, col)

    def unify_column(self, col: int) -> None:
        """
        convert all the values of a column into one, adjust ranges
        """
        self.__require_normalized()
        col = self.__normalize_col_num(col)
        minv, maxv = self.__get_col_range(col)
        diff = maxv - minv - 1
        for i in range(col + 1, len(self.idrange)):
            self.idrange[i] -= diff
            self.__interactions[:, i] -= diff
        self.__interactions[:, col] = minv
        self.idrange[col] = minv + 1

    def get_counts(self):
        """
        return an array of interactions size with the amount of same user, item interactions for every pair
        """
        counts = {}
        col = np.zeros(self.__interactions.shape[0])
        for i, row in enumerate(self.__interactions):
            key = row[0:2]
            k = tuple(key)
            if k in counts:
                v = counts[k]
            else:
                v = np.count_nonzero((self.__interactions[:, 0:2] == key).all(axis=1))
                counts[k] = v
            col[i] = v
        return col

    def __swap_columns(self, col1: int, col2: int) -> None:
        assert col1 < col2
        min1, max1 = self.__get_col_range(col1)
        min2, max2 = self.__get_col_range(col2)
        values1 = self.__interactions[:, col1]
        values2 = self.__interactions[:, col2].copy()
        colmapping1 = np.arange(min1, max1)
        colmapping2 = np.arange(min2, max2)
        self.remove_column(col1)
        self.insert_column(col1, values2, colmapping2)
        self.insert_column(col2, values1, colmapping1)
        self.remove_column(col2 + 1)

    def swap_columns(self, col1: int, col2: int) -> None:
        """
        swap two dataset columns maintaining the ranges
        """
        col1 = self.__normalize_col_num(col1)
        col2 = self.__normalize_col_num(col2)
        if col1 == col2:
            return
        if col1 > col2:
            return self.__swap_columns(col2, col1)
        return self.__swap_columns(col1, col2)

    def __shift_column_values(self, col: int) -> int:
        minv = 0 if col == 0 else self.idrange[col - 1]
        cond = self.__interactions[:, col] != minv
        c = len(self.__interactions)
        self.__interactions =  self.__interactions[cond]
        c -= len(self.__interactions)
        for i in range(col, len(self.idrange)):
            self.idrange[i] -= 1
            self.__interactions[:, i] -= 1
        return c

    def prepare_for_recommend(self, prev_item_col: int=None) -> int:
        """
        prepare dataset for recommendation task:
        instead of given a user get the rating for an item
        given a previous item get the rating for an item
        """
        self.__require_normalized()
        if prev_item_col is None:
            self.add_previous_item_column()
            prev_item_col = -2
        prev_item_col = self.__normalize_col_num(prev_item_col)
        assert prev_item_col > 1 and prev_item_col < len(self.idrange)
        self.remove_column(0) # remove users
        # swap items and previous items
        item_col = 0
        prev_item_col -= 1
        self.__swap_columns(item_col, prev_item_col)
        # remove previous items 0 (no previous item)
        return self.__shift_column_values(item_col)


class InteractionPairDataset(torch.utils.data.Dataset):
    """
    returns pairs of positive and negative interactions
    used in the trainset when pairwise loss is enabled
    """

    def __init__(self, dataset: Container[np.ndarray], groups: Container[np.ndarray]):
        self.dataset = dataset
        self.indices = []
        for group in groups:
            if len(group) < 2:
                continue
            p = group[0]
            for n in group[1:]:
                self.indices.append((p, n))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index) -> Tuple:
        p, n = self.indices[index]
        pos, neg = self.dataset[p], self.dataset[n]
        assert pos[0] == neg[0]
        return pos, neg


class GroupingDataset(torch.utils.data.Dataset):
    """
    groups the data by values in one column
    used for the testset to get batches separated by user
    """

    def __init__(self, dataset: Container[np.ndarray], groups: Container[np.ndarray]):
        self.dataset = dataset
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index) -> Tuple:
        return self.dataset[self.groups[index]]


def vstack_collate_fn(batch: Container[np.ndarray]):
    """
    used in a DataLoader.collate_fn to stack the batch vertically
    """
    return torch.from_numpy(np.vstack(batch))


class BaseDatasetSource:
    """
    basic class to load a dataset
    both trainset and testset should have the following structure
    (user id, item id, rating, context 1, context 2)

    user id, item id and context columns should be normalized,
    meaning that the values are consecutive and don't overlap

    items should be a pandas Dataset indexed by item_id
    """

    def __init__(self, logger: Logger=None):
        self._logger = logger or get_logger(self)
        self.trainset: InteractionDataset = None
        self.testset: InteractionDataset = None
        self.useritems: sp.spmatrix = None
        self.items: DataFrame = None

    def load(self, hparams: HyperParameters):
        raise NotImplementedError()

    def _setup(self, hparams: HyperParameters, min_item_interactions: int=0, min_user_interactions: int=0, previous_item_col: int=None) -> Container[np.ndarray]:

        self._logger.info("normalizing ids...")
        mapping = self.trainset.normalize_ids()
        self._logger.info("calculating user-item matrix...")
        self.useritems = self.trainset.create_adjacency_submatrix()

        remove = min_item_interactions > 0 or min_user_interactions > 0
        if remove:
            self._logger.info("removing low interactions...")
            ci = self.trainset.remove_low_items(self.useritems, min_item_interactions)
            if ci > 0:
                self._logger.info(f"removed {ci} interactions of items with less than {min_item_interactions} users")
            cu = self.trainset.remove_low_users(self.useritems, min_user_interactions)
            if cu > 0:
                self._logger.info(f"removed {cu} interactions of users with less than {min_user_interactions} items")
            if (cu > 0 or ci > 0):
                self._logger.info("normalizing ids again...")
                self.trainset.denormalize_ids(mapping)
                mapping = self.trainset.normalize_ids()
                self._logger.info("calculating user-item matrix again...")
                self.useritems = self.trainset.create_adjacency_submatrix()

        if previous_item_col is None and hparams.should_have_interaction_context("previous"):
            self._logger.info("adding previous item column...")
            self.trainset.add_previous_item_column()
            previous_item_col = -2

        if hparams.interaction_context == "random":
            self._logger.info("adding random item column...")
            isize = self.trainset.idrange[1] - self.trainset.idrange[0]
            self.trainset.add_random_column(isize)

        if hparams.recommend:
            self._logger.info("preparing for recommend...")
            cr = self.trainset.prepare_for_recommend(previous_item_col)
            if cr > 0:
                self._logger.info(f"removed {cr} interactions without previous item")

        if hparams.max_interactions > 0:
            trainlen = len(self.trainset)
            if hparams.max_interactions < trainlen:
                self._logger.info(f"reducing from {trainlen} to {hparams.max_interactions} interactions...")
                self.trainset.remove_random(hparams.max_interactions)

        self._logger.info("extracting test dataset...")
        self.testset = self.trainset.extract_test_dataset()
        return mapping


def save_model(path: str, model, idrange: np.ndarray, items: DataFrame = None):
    data = {
        "model": model,
        "idrange": idrange,
        "items": items
    }
    with open(path, "wb") as fh:
        torch.save(data, fh)


def load_model(path: str):
    with open(path, "rb") as fh:
        data = torch.load(fh)
        return data["model"], data["idrange"], data["items"]


class IdFinder:
    """
    given a container with ordered ids,
    this class uses bisect to find the position of one.
    It's useful to convert non-consecutive ids in a dataset
    into consecutive integers.
    """

    def __init__(self, data: Container=[], hash: bool=False):
        self.data = data
        self.__hash = hash

    def __len__(self):
        return len(self.data)

    def _fix(self, v):
        if self.__hash:
            v = hash(v)
        elif not isinstance(v, int):
            v = int(v)
        return v

    def find(self, v) -> int:
        v = self._fix(v)
        id = bisect_left(self.data, v)
        if not self._check(id, v):
            return None
        return id

    def _check(self, id, v):
        return id >= 0 and id < len(self.data) and self.data[id] == v

    def reverse(self, v: int) -> Any:
        v = int(v)
        if v < 0 or v >= len(self.data):
            return None
        return self.data[v]


class IdGenerator(IdFinder):
    """
    additional methods to add the ids
    """

    def __init__(self, hash: bool=False):
        super().__init__([], hash)

    def add(self, v) -> None:
        v = self._fix(v)
        id = bisect_left(self.data, v)
        if not self._check(id, v):
            self.data.insert(id, v)
