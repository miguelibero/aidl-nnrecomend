import sqlite3
import numpy as np
import hashlib
from nnrecommend.dataset import Dataset
from nnrecommend.logging import get_logger
from logging import Logger


class ItunesPodcastsDataset:
    """
    the dataset can be downloaded from https://www.kaggle.com/thoughtvector/podcastreviews
    """
    def __init__(self, path: str, logger: Logger=None, maxsize: int=-1):
        self.__path = path
        self.__logger = logger or get_logger(self)
        self.__maxsize = maxsize
        self.trainset = None
        self.testset = None
        self.matrix = None

    COND = "WHERE rating == 5"
    ROW_LOAD_PRINT_STEP = 2

    def load(self) -> None:
        self.__logger.info("loading database...")
        con = sqlite3.connect(self.__path)
        cur = con.cursor()
        cur.execute(f'SELECT COUNT(*) FROM reviews {self.COND}')
        size = cur.fetchone()[0]
        if self.__maxsize > 0 and self.__maxsize < size:
            size = self.__maxsize
        self.__logger.info(f"loading {size} reviews...")
        interactions = np.zeros((size, 2), dtype=int)
        r = cur.execute(f'SELECT author_id, podcast_id FROM reviews {self.COND} LIMIT ?', (size,))
        users = []
        items = []
        lp = 0

        def get_id(v, elements):
            try:
                return users.index(v)
            except:
                users.append(v)
                return len(users) - 1

        for i, row in enumerate(r):
            u = get_id(row[0], users)
            v = get_id(row[1], items)
            p = 100*i/size
            if p - lp > self.ROW_LOAD_PRINT_STEP:
                lp = p
                self.__logger.debug(f"loading rows... {p:.2f}%")
            interactions[i] = (u, v)
        self.__logger.info("setting up datasets..")
        self.trainset = Dataset(interactions)
        self.__logger.info("normalizing dataset ids..")
        self.trainset.normalize_ids()
        self.__logger.info("extracting test dataset..")
        self.testset = self.trainset.extract_test_dataset()
        self.__logger.info("calculating adjacency matrix..")
        self.matrix = self.trainset.create_adjacency_matrix()
        
    def setup(self, negatives_train: int, negatives_test: int) -> None:
        if self.trainset is None:
            self.load()
        self.__logger.info("adding negative sampling...")
        self.trainset.add_negative_sampling(self.matrix, negatives_train)
        self.testset.add_negative_sampling(self.matrix, negatives_test)