import pandas as pd
from nnrecommend.dataset import Dataset
from nnrecommend.logging import get_logger
from logging import Logger

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
        self.features = None


    def load(self, maxsize: int=-1) -> None:
        self.__logger.info("loading training dataset...")
        self.trainset = Dataset(pd.read_csv(f"{self.__path}.train.rating", sep='\t', header=None))
        iddiff = self.trainset.normalize_ids()
        self.__logger.info("loading test dataset...")
        self.testset = Dataset(pd.read_csv(f"{self.__path}.test.rating", sep='\t', header=None))
        self.testset.normalize_ids(iddiff)
        self.__logger.info("calculating adjacency matrix...")
        self.matrix = self.trainset.create_adjacency_matrix()