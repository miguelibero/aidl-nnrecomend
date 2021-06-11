from logging import Logger
import sqlite3
from sqlite3.dbapi2 import Cursor
from typing import Container, Dict
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

    def __load_reviews(self, cur: Cursor, maxsize: int) -> None:
        cur.execute(f'SELECT COUNT(*) FROM reviews {self.COND}')
        size = cur.fetchone()[0]
        if maxsize > 0 and maxsize < size:
            size = maxsize
        self._logger.info(f"loading {size} reviews...")
        r = cur.execute(f'SELECT author_id, podcast_id FROM reviews {self.COND} LIMIT ?', (size,))
        return r, size

    def __load_item_info(self, cur: Cursor, items: Container[str]) -> Dict[str, Dict[str, str]]:
        info = {}
        r = cur.execute(f'SELECT podcast_id, itunes_url, title FROM podcasts')
        for row in r:
            info[row[0]] = {
                'url': row[1],
                'title': row[2]
            }
        self._logger.info(f"loaded info for {len(info)} podcasts")
        return info

    def __fix_item_info(self, info: Dict[str, Dict[str, str]], items: Container[str], mapping: np.ndarray) -> Dict[int, Dict[str, str]] :

        def find(container, val):
            idx = bisect_left(container, val)
            if idx < 0 or idx >= len(container) or container[idx] != val:
                return -1
            return idx

        finfo = {}
        for k, elm in info.items():
            i = find(items, hash(k))
            if i < 0:
                continue
            if mapping is not None:
                i = find(mapping, i)
                if i < 0:
                    continue
            finfo[i] = elm
        return finfo

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
        return interactions, items

    def load(self, maxsize: int=-1) -> None:

        with sqlite3.connect(self.__path) as con:
            cur = con.cursor()
            self._logger.info("loading reviews...")
            reviews, size = self.__load_reviews(cur, maxsize)
            self._logger.info("generating interactions...")
            interactions, items = self.__generate_interactions(reviews, size)
            self._logger.info("loading item info...")
            item_info = self.__load_item_info(cur, items)

        self._logger.info("setting up datasets..")
        self.trainset = Dataset(interactions)
        self._logger.info("normalizing dataset ids..")
        mapping = self.trainset.normalize_ids()
        self._logger.info("fixing item info..")
        self.item_info = self.__fix_item_info(item_info, items, mapping[1])
        self._logger.info("extracting test dataset..")
        self.testset = self.trainset.extract_test_dataset()
        self._logger.info("calculating adjacency matrix..")
        self.matrix = self.trainset.create_adjacency_matrix()