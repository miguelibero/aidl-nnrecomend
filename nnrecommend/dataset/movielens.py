from logging import Logger
from typing import Dict
import pandas as pd
import numpy as np
import os
from nnrecommend.hparams import HyperParameters
from nnrecommend.dataset import IdFinder, InteractionDataset, BaseDatasetSource


COLUMN_NAMES = ('user_id', 'item_id', 'label', 'timestamp')
LOAD_COLUMNS = ('user_id', 'item_id', 'label')
ITEMINFO_COLUMN_NAMES = ("item_id", "title", "release_date", "unknown", "link")
ITEMINFO_LOAD_COLUMNS = ("title", "release_date", "link")

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
        self._logger.info("normalizing dataset ids..")
        mapping = self.trainset.normalize_ids(assume_consecutive=True)
        self._logger.info("loading test dataset...")
        self.testset = InteractionDataset(self.__load_data("test", maxsize))
        self.testset.map_ids(mapping)
        self._logger.info("calculating adjacency matrix...")
        self.matrix = self.trainset.create_adjacency_matrix()


class Movielens100kDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://www.kaggle.com/prajitdatta/movielens-100k-dataset/
    """

    INTERACTIONS_FILE = "u.data"
    ITEMINFO_FILE = "u.item"

    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    def __load_interactions(self, maxsize: int):
        nrows = maxsize if maxsize > 0 else None
        path = os.path.join(self.__path, self.INTERACTIONS_FILE)
        data = pd.read_csv(path, sep='\t', header=None, nrows=nrows, names=COLUMN_NAMES)
        data.sort_values(by=['user_id', 'timestamp'], inplace=True, ascending=True)
        data = np.array(data[[*LOAD_COLUMNS]], dtype=np.int64)
        data[:, 2] = 1
        return data

    def __load_iteminfo(self, mapping: np.ndarray) -> Dict[int, Dict[str, str]]:
        path = os.path.join(self.__path, self.ITEMINFO_FILE)
        data = pd.read_csv(path, index_col=False, sep='|', dtype=str, header=None, names=ITEMINFO_COLUMN_NAMES)
        data = data[[*ITEMINFO_LOAD_COLUMNS]]
        infos = {}
        mapping = None if mapping is None else IdFinder(mapping)
        for i, row in data.iterrows():
            if mapping:
                i = mapping.find(i)
            if i >= 0:
                infos[i] = row.to_dict()
        return infos

    def load(self, hparams: HyperParameters) -> None:
        maxsize = hparams.max_interactions
        self._logger.info("loading training dataset...")
        self.trainset = InteractionDataset(self.__load_interactions(maxsize))
        self._logger.info("normalizing dataset ids..")
        mapping = self.trainset.normalize_ids()
        if hparams.should_have_interaction_context("previous"):
            self._logger.info("adding previous item column...")
            self.trainset.add_previous_item_column()
        self._logger.info("extracting test dataset..")
        self.testset = self.trainset.extract_test_dataset()
        self._logger.info("calculating adjacency matrix..")
        self.matrix = self.trainset.create_adjacency_matrix()
        self.iteminfo = self.__load_iteminfo(mapping[0])