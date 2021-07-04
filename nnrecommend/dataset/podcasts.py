import sqlite3
import numpy as np
import pandas as pd
from logging import Logger
from sqlite3.dbapi2 import Connection
from typing import Container, Dict
from pandas.core.frame import DataFrame
from nnrecommend.hparams import HyperParameters
from nnrecommend.dataset import BaseDatasetSource, IdFinder, InteractionDataset


class ItunesPodcastsDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://www.kaggle.com/thoughtvector/podcastreviews
    """
    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    REVIEWS_QUERY = 'SELECT podcast_id, author_id FROM reviews WHERE rating == 5 ORDER BY created_at ASC LIMIT :limit'

    def __load_reviews(self, conn: Connection, maxsize: int) -> None:
        data = pd.read_sql(self.REVIEWS_QUERY, conn, params={'limit': maxsize})
        for colname in data.select_dtypes(exclude=int):
            data[colname] = data[colname].apply(hash)
        return data

    PODCASTS_QUERY = 'SELECT podcast_id, itunes_url, title FROM podcasts'
    PODCASTS_INDEX = 'podcast_id'

    def __load_iteminfo(self, conn: Connection) -> DataFrame:
        data = pd.read_sql(self.PODCASTS_QUERY, conn, index_col=self.PODCASTS_INDEX)
        data[self.PODCASTS_INDEX] = data[self.PODCASTS_INDEX].apply(hash)
        self._logger.info(f"loaded info for {len(data)} podcasts")
        return data

    def __fix_iteminfo(self, data: DataFrame, items: Container[str], mapping: np.ndarray) -> Dict[int, Dict[str, str]] :
        items = IdFinder(items)
        mapping = None if mapping is None else IdFinder(mapping)

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

    def load(self, hparams: HyperParameters) -> None:
        with sqlite3.connect(self.__path) as conn:
            interactions = self.__load_reviews(conn, hparams.max_interactions)
            self.trainset = InteractionDataset(interactions)

        self._setup(hparams, 1, 1)


    def load_recommend(self, hparams: HyperParameters):
        with sqlite3.connect(self.__path) as conn:
            interactions = self.__load_reviews(conn, hparams.max_interactions)
            self.trainset = InteractionDataset(interactions)
            self._logger.info("loading item info...")
            iteminfo = self.__load_iteminfo(conn)

        mapping = self._setup(hparams, 1, 1)
        self._logger.info("fixing item info..")
        self.iteminfo = self.__fix_iteminfo(iteminfo, mapping[1])