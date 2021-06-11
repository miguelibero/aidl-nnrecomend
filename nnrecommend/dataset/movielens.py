from logging import Logger
import pandas as pd
from nnrecommend.dataset import Dataset, BaseDatasetSource
import numpy as np

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
        data = pd.read_csv(path, sep='\t', header=None, nrows=nrows)
        return np.array(data, dtype=np.int64)[:, :2]

    def load(self, maxsize: int=-1) -> None:
        self._logger.info("loading training dataset...")
        self.trainset = Dataset(self.__load_data("train", maxsize))
        mapping = self.trainset.normalize_ids()
        self._logger.info("loading test dataset...")
        self.testset = Dataset(self.__load_data("test", maxsize))
        self.testset.normalize_ids(mapping)
        self._logger.info("calculating adjacency matrix...")
        self.matrix = self.trainset.create_adjacency_matrix()


class Movielens100kDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://www.kaggle.com/prajitdatta/movielens-100k-dataset/
    """
    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    def __load_data(self, type:str, maxsize: int):
        nrows = maxsize if maxsize > 0 else None
        path = self.__path
        data = pd.read_csv(path, sep='\t', header=None, nrows=nrows)
        data = np.array(data, dtype=np.int64)[:, :2]
        data[:, 2] = 1
        return data

    def load(self, maxsize: int=-1) -> None:
        self._logger.info("loading training dataset...")
        self.trainset = Dataset(self.__load_data("train", maxsize))
        self._logger.info("normalizing dataset ids..")
        self.trainset.normalize_ids()
        self._logger.info("extracting test dataset..")
        self.testset = self.trainset.extract_test_dataset()
        self._logger.info("calculating adjacency matrix..")
        self.matrix = self.trainset.create_adjacency_matrix()