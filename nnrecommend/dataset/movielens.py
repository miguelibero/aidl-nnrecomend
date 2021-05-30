import pandas as pd
from nnrecommend.dataset import Dataset
from nnrecommend.logging import get_logger
from logging import Logger
import numpy as np

class MovielensDataset:
    """
    the dataset can be downloaded from https://drive.google.com/uc?id=1rE20sLow9sT2ULpBOOWqw2SEnpIm16OZ
    """
    def __init__(self, path: str, logger: Logger=None):
        self.__path = path
        self.__logger = logger or get_logger(self)
        self.trainset = None
        self.testset = None
        self.matrix = None

    def __load_data(self, type:str, maxsize: int):
        nrows = maxsize if maxsize > 0 else None
        path = f"{self.__path}.{type}.rating"
        data = pd.read_csv(path, sep='\t', header=None, nrows=nrows)
        return np.array(data, dtype=np.int64)

    def load(self, maxsize: int=-1) -> None:
        self.__logger.info("loading training dataset...")
        self.trainset = Dataset(self.__load_data("train", maxsize))
        iddiff = self.trainset.normalize_ids()
        self.__logger.info("loading test dataset...")
        self.testset = Dataset(self.__load_data("test", maxsize))
        self.testset.normalize_ids(iddiff)
        self.__logger.info("calculating adjacency matrix...")
        self.matrix = self.trainset.create_adjacency_matrix()