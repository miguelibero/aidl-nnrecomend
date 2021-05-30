from logging import Logger
import sqlite3
import numpy as np
from nnrecommend.dataset import BaseDatasetSource, Dataset
from bisect import bisect_left


class ItunesPodcastsDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://www.kaggle.com/thoughtvector/podcastreviews
    """
    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    COND = "WHERE rating == 5"
    ROW_LOAD_PRINT_STEP = 2

    def __load_database(self, maxsize: int) -> None:
        con = sqlite3.connect(self.__path)
        cur = con.cursor()
        cur.execute(f'SELECT COUNT(*) FROM reviews {self.COND}')
        size = cur.fetchone()[0]
        if maxsize > 0 and maxsize < size:
            size = maxsize
        self._logger.info(f"loading {size} reviews...")
        r = cur.execute(f'SELECT author_id, podcast_id FROM reviews {self.COND} LIMIT ?', (size,))
        return r, size

    def __generate_interactions(self, rows, size):
        interactions = np.zeros((size, 2), dtype=int)
        users = []
        items = []
        lp = 0

        def get_id(v, elements):
            v = hash(v)
            i = bisect_left(elements, v)
            elements.insert(i, v)
            return i

        for i, row in enumerate(rows):
            u = get_id(row[0], users)
            v = get_id(row[1], items)
            p = 100*i/size
            if p - lp > self.ROW_LOAD_PRINT_STEP:
                lp = p
                self._logger.info(f"{p:.2f}%")
            interactions[i] = (u, v)
        return interactions

    def load(self, maxsize: int=-1) -> None:
        self._logger.info("loading database...")
        rows, size = self.__load_database(maxsize)
        self._logger.info("generating interactions...")
        interactions = self.__generate_interactions(rows, size)
        self._logger.info("setting up datasets..")
        self.trainset = Dataset(interactions)
        self._logger.info("normalizing dataset ids..")
        self.trainset.normalize_ids()
        self._logger.info("extracting test dataset..")
        self.testset = self.trainset.extract_test_dataset()
        self._logger.info("calculating adjacency matrix..")
        self.matrix = self.trainset.create_adjacency_matrix()