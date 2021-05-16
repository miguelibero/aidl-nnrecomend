from typing import Callable
from nnrecommend.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd


class MovielensData:

    def __init__(self, logger: Callable[[str], None]=None):
        self.__logger = logger

    def _log(self, msg):
        if self.__logger is not None:
            self.__logger(msg)

    def load(self, path: str) -> None:
        self._log("loading datasets...")
        self.dataset = Dataset(pd.read_csv(f"{path}.train.rating", sep='\t', header=None))
        iddiff = self.dataset.normalize_ids()
        self.testset = Dataset(pd.read_csv(f"{path}.test.rating", sep='\t', header=None))
        self.testset.normalize_ids(iddiff)
        self._log("calculating adjacency matrix...")
        self.matrix = self.dataset.create_adjacency_matrix()
    
    def setup(self, batch_size: int, negatives_train: int, negatives_test: int) -> None:
        self._log("adding negative sampling...")
        self.dataset.add_negative_sampling(self.matrix, negatives_train)
        self.testset.add_negative_sampling(self.matrix, negatives_test)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size)
        self.testloader = DataLoader(self.testset, batch_size=negatives_test+1)