import os
import pandas as pd
import numpy as np
from logging import Logger
from pandas.core.frame import DataFrame
from nnrecommend.hparams import HyperParameters
from nnrecommend.dataset import IdFinder, InteractionDataset, BaseDatasetSource


class MovielensLabDatasetSource(BaseDatasetSource):
    """
    this loads the dataset provided in the recommender systems laboratory
    the dataset can be downloaded from https://drive.google.com/uc?id=1rE20sLow9sT2ULpBOOWqw2SEnpIm16OZ
    """
    COLUMN_NAMES = ('user_id', 'item_id', 'label', None)

    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    def __load_data(self, type:str):
        path = f"{self.__path}.{type}.rating"
        data = pd.read_csv(path, sep='\t', header=None, names=self.COLUMN_NAMES)
        data = data.drop([None], axis=1)
        return data

    def load(self, hparams: HyperParameters) -> None:
        self._logger.info("loading training dataset...")
        interactions = self.__load_data("train")
        self.trainset = InteractionDataset(interactions)
        self._logger.info("loading testing dataset...")
        interactions = self.__load_data("test")
        self.testset = InteractionDataset(interactions)
        self._logger.info("normalizing ids...")
        mapping = self.trainset.normalize_ids(assume_consecutive=True)
        self._logger.info("calculating user-item matrix...")
        self.useritems = self.trainset.create_adjacency_submatrix()
        self.testset.map_ids(mapping)


class Movielens100kDatasetSource(BaseDatasetSource):
    """
    this loads the raw Movielens 100k dataset
    the dataset can be downloaded from https://www.kaggle.com/prajitdatta/movielens-100k-dataset/
    """

    INTERACTIONS_FILE = "u.data"
    ITEMINFO_FILE = "u.item"

    LABEL_COLUMN = 'label'
    SORT_COLUMN = 'timestamp'
    USER_COLUMN = 'user_id'
    ITEM_COLUMN = 'item_id'
    ORIGINAL_ITEM_ID_COLUMN = "original_item_id"
    COLUMNS = (USER_COLUMN, ITEM_COLUMN, LABEL_COLUMN, SORT_COLUMN)
    SORT_COLUMNS = (USER_COLUMN, SORT_COLUMN)

    ITEM_ID_COLUMN = ITEM_COLUMN
    ITEM_COLUMNS = (ITEM_ID_COLUMN, "title", "release_date", None, "link")

    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    def __load_interactions(self):
        path = os.path.join(self.__path, self.INTERACTIONS_FILE)
        data = pd.read_csv(path, sep='\t', header=None, names=self.COLUMNS)
        data.sort_values(by=list(self.SORT_COLUMNS), inplace=True, ascending=True)
        del data[self.SORT_COLUMN]
        del data[self.LABEL_COLUMN]
        return data

    def __load_items(self, mapping: np.ndarray) -> DataFrame:
        path = os.path.join(self.__path, self.ITEMINFO_FILE)
        data = pd.read_csv(path, index_col=False, sep='|', dtype=str, header=None,
            names=self.ITEM_COLUMNS)
        data = data.drop([None], axis=1)
        mapping = IdFinder(mapping)
        data[self.ORIGINAL_ITEM_ID_COLUMN] = data[self.ITEM_ID_COLUMN].copy()
        data[self.ITEM_ID_COLUMN] = data[self.ITEM_ID_COLUMN].apply(mapping.find)
        data.dropna(subset=[self.ITEM_ID_COLUMN], inplace=True)
        data.set_index(self.ITEM_ID_COLUMN, inplace=True)
        self._logger.info(f"loaded info for {len(data)} movies")
        return data

    def load(self, hparams: HyperParameters) -> None:
        self._logger.info("loading interactions...")
        interactions = self.__load_interactions()
        self.trainset = InteractionDataset(interactions, add_labels_col=True)
        mapping = self._setup(hparams)
        self._logger.info("loading movies...")
        self.items = self.__load_items(mapping[1])