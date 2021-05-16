from typing import Callable
from nnrecommend.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd


class MovielensDataset:
    """
    the dataset can be downloaded from https://drive.google.com/uc?id=1rE20sLow9sT2ULpBOOWqw2SEnpIm16OZ
    """
    def __init__(self, path: str, logger: Callable[[str], None]=None):
        self.__path = path
        self.__logger = logger
        self.trainset = None
        self.testset = None
        self.matrix = None
        self.trainloader = None
        self.testloader = None

    def _log(self, msg):
        if self.__logger is not None:
            self.__logger(msg)

    def load(self) -> None:
        self._log("loading training dataset...")
        self.trainset = Dataset(pd.read_csv(f"{self.__path}.train.rating", sep='\t', header=None))
        iddiff = self.trainset.normalize_ids()
        self._log("loading test dataset...")
        self.testset = Dataset(pd.read_csv(f"{self.__path}.test.rating", sep='\t', header=None))
        self.testset.normalize_ids(iddiff)
        self._log("calculating adjacency matrix...")
        self.matrix = self.trainset.create_adjacency_matrix()
    
    def setup(self, negatives_train: int, negatives_test: int) -> None:
        if self.trainset is None:
            self.load()
        self._log("adding negative sampling...")
        self.trainset.add_negative_sampling(self.matrix, negatives_train)
        self.testset.add_negative_sampling(self.matrix, negatives_test)