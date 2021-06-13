from logging import Logger
from nnrecommend.hparams import HyperParameters
import sqlite3
from sqlite3.dbapi2 import Cursor
from typing import Container, Dict
import numpy as np
from nnrecommend.dataset import BaseDatasetSource, Dataset, IdGenerator
from bisect import bisect_left


class ItunesPodcastsDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://www.kaggle.com/thoughtvector/podcastreviews
    """
    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    COND = "WHERE rating == 5"

    def __load_reviews(self, cur: Cursor, maxsize: int) -> None:
        cur.execute(f'SELECT COUNT(*) FROM reviews {self.COND}')
        size = cur.fetchone()[0]
        if maxsize > 0 and maxsize < size:
            size = maxsize
        self._logger.info(f"loading {size} reviews...")
        r = cur.execute(f'SELECT author_id, podcast_id FROM reviews {self.COND} ORDER BY created_at ASC LIMIT ?', (size,))
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

        items = IdGenerator(items)
        mapping = None if mapping is None else IdGenerator(mapping)

        finfo = {}
        for k, elm in info.items():
            i = items.find(k)
            if i < 0:
                continue
            if mapping is not None:
                i = mapping.find(i)
                if i < 0:
                    continue
            finfo[i] = elm
        return finfo

    def __generate_interactions(self, rows, size):
        users = IdGenerator()
        items = IdGenerator()

        self._logger.info("generating user and item ids...")
        data = []
        for row in rows:
            users.add(row[0])
            items.add(row[1])
            data.append((row[0], row[1]))

        self._logger.info("finding user and item ids...")
        interactions = np.zeros((size, 2), dtype=int)
        for i, row in enumerate(data):
            interactions[i][:2] = (
                users.find(row[0]),
                items.find(row[1])
            )
        return interactions, items.data

    def load(self, hparams: HyperParameters) -> None:
        maxsize = hparams.max_interactions

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
        self._logger.info("calculating adjacency matrix..")
        self.matrix = self.trainset.create_adjacency_matrix()
        self._logger.info("removing low interactions...")
        ci = self.trainset.remove_low_items(self.matrix, 1)
        cu = self.trainset.remove_low_users(self.matrix, 1)
        recalc = cu > 0 or ci > 0
        if recalc:
            self._logger.info(f"removed {cu} users and {ci} items")
            self._logger.info("normalizing ids again...")
            self.trainset.denormalize_ids(mapping)
            mapping = self.trainset.normalize_ids()
        if hparams.should_have_interaction_context(0):
            self._logger.info("adding previous item column...")
            self.trainset.add_previous_item_column()
        if recalc:
            self._logger.info("calculating adjacency matrix again...")
            self.matrix = self.trainset.create_adjacency_matrix()
        self._logger.info("extracting test dataset..")
        self.testset = self.trainset.extract_test_dataset()
        self._logger.info("fixing item info..")
        self.item_info = self.__fix_item_info(item_info, items, mapping[1])
