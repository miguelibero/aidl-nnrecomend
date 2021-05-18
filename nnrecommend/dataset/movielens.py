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
        self.trainloader = None
        self.testloader = None

    def load(self) -> None:
        self.__logger.info("loading training dataset...")
        self.trainset = Dataset(pd.read_csv(f"{self.__path}.train.rating", sep='\t', header=None))
        iddiff = self.trainset.normalize_ids()
        self.__logger.info("loading test dataset...")
        self.testset = Dataset(pd.read_csv(f"{self.__path}.test.rating", sep='\t', header=None))
        self.testset.normalize_ids(iddiff)
        self.__logger.info("calculating adjacency matrix...")
        self.matrix = self.trainset.create_adjacency_matrix()
    
    def setup(self, negatives_train: int, negatives_test: int) -> None:
        if self.trainset is None:
            self.load()
        self.__logger.info("adding negative sampling...")
        self.trainset.add_negative_sampling(self.matrix, negatives_train)
        self.testset.add_negative_sampling(self.matrix, negatives_test)