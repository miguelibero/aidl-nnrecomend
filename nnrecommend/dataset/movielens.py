from logging import Logger
import pandas as pd
import numpy as np
import os
from pandas.core.frame import DataFrame
from nnrecommend.hparams import HyperParameters
from nnrecommend.dataset import IdFinder, InteractionDataset, BaseDatasetSource


COLUMN_NAMES = ('user_id', 'item_id', 'label', 'timestamp')
LOAD_COLUMNS = ('user_id', 'item_id', 'label')


class MovielensLabDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://drive.google.com/uc?id=1rE20sLow9sT2ULpBOOWqw2SEnpIm16OZ
    """
    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    def __load_data(self, type:str, maxsize: int):
        nrows = maxsize if maxsize > 0 else None
        path = f"{self.__path}.{type}.rating"
        data = pd.read_csv(path, sep='\t', header=None, nrows=nrows, names=COLUMN_NAMES)
        return data[[*LOAD_COLUMNS]]

    def load(self, hparams: HyperParameters) -> None:
        maxsize = hparams.max_interactions
        self._logger.info("loading training dataset...")
        self.trainset = InteractionDataset(self.__load_data("train", maxsize))
        self._setup(hparams)


class Movielens100kDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://www.kaggle.com/prajitdatta/movielens-100k-dataset/
    """

    INTERACTIONS_FILE = "u.data"
    ITEMINFO_FILE = "u.item"

    SORT_COLUMNS = ('user_id', 'timestamp')
    ITEM_COLUMN_NAMES = ("item_id", "title", "release_date", None, "link")
    ITEM_INDEX_COL = "item_id"

    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    def __load_interactions(self, maxsize: int):
        nrows = maxsize if maxsize > 0 else None
        path = os.path.join(self.__path, self.INTERACTIONS_FILE)
        data = pd.read_csv(path, sep='\t', header=None, nrows=nrows, names=COLUMN_NAMES)
        data.sort_values(by=list(self.SORT_COLUMNS), inplace=True, ascending=True)
        data = np.array(data[[*LOAD_COLUMNS]], dtype=np.int64)
        data[:, 2] = 1
        return data

    def __load_items(self, mapping: np.ndarray) -> DataFrame:
        path = os.path.join(self.__path, self.ITEMINFO_FILE)
        data = pd.read_csv(path, index_col=False, sep='|', dtype=str, header=None,
            names=self.ITEM_COLUMN_NAMES)
        mapping = IdFinder(mapping)
        data[self.ITEM_INDEX_COL] = data[self.ITEM_INDEX_COL].apply(mapping.find)
        data.dropna(subset=[self.ITEM_INDEX_COL], inplace=True)
        data.set_index(self.ITEM_INDEX_COL, inplace=True)
        self._logger.info(f"loaded info for {len(data)} movies")
        return data

    def load(self, hparams: HyperParameters) -> None:
        self._logger.info("loading training dataset...")
        data = self.__load_interactions(hparams.max_interactions)
        self.trainset = InteractionDataset(data)
        mapping = self._setup(hparams)
        if hparams.recommend:
            self._logger.info("loading movies...")
            self.items = self.__load_items(mapping[1])