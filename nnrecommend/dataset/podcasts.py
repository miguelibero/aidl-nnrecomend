import sqlite3
import numpy as np
import pandas as pd
from logging import Logger
from sqlite3.dbapi2 import Connection
from typing import Dict
from pandas.core.frame import DataFrame
from nnrecommend.hparams import HyperParameters
from nnrecommend.dataset import BaseDatasetSource, IdFinder, InteractionDataset


MIN_ITEM_INTERACTIONS = 1
MIN_USER_INTERACTIONS = 3


class ItunesPodcastsDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://www.kaggle.com/thoughtvector/podcastreviews
    """
    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    INTERACTIONS_QUERY = 'SELECT author_id, podcast_id FROM reviews WHERE rating == 5 ORDER BY created_at ASC LIMIT :limit'

    def __load_interactions(self, conn: Connection, maxsize: int) -> None:
        data = pd.read_sql(self.INTERACTIONS_QUERY, conn, params={'limit': maxsize})
        for colname in data.select_dtypes(exclude=int):
            data[colname] = data[colname].apply(hash)
        return data

    ITEMS_QUERY = 'SELECT podcast_id, itunes_url, title FROM podcasts'
    ITEM_INDEX_COL = 'podcast_id'

    def __load_items(self, conn: Connection) -> DataFrame:
        data = pd.read_sql(self.ITEMS_QUERY, conn)
        data["original_podcast_id"] = data[self.ITEM_INDEX_COL].copy()
        data[self.ITEM_INDEX_COL] = data[self.ITEM_INDEX_COL].apply(hash)
        self._logger.info(f"loaded info for {len(data)} podcasts")
        return data

    def __fix_items(self, data: DataFrame, mapping: np.ndarray) -> Dict[int, Dict[str, str]] :
        mapping = IdFinder(mapping)
        data[self.ITEM_INDEX_COL] = data[self.ITEM_INDEX_COL].apply(mapping.find)
        data.dropna(subset=[self.ITEM_INDEX_COL], inplace=True)
        data.set_index(self.ITEM_INDEX_COL, inplace=True)
        self._logger.info(f"valid info for {len(data)} podcasts")
        return data

    def load(self, hparams: HyperParameters) -> None:
        with sqlite3.connect(self.__path) as conn:
            interactions = self.__load_interactions(conn, hparams.max_interactions)
            self.trainset = InteractionDataset(interactions)
            if hparams.recommend:
                self._logger.info("loading items...")
                items = self.__load_items(conn)
        mapping = self._setup(hparams, MIN_ITEM_INTERACTIONS, MIN_USER_INTERACTIONS)
        if hparams.recommend:
            self._logger.info("fixing items...")
            self.items = self.__fix_items(items, mapping[1])

