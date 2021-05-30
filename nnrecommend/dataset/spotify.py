import pandas as pd
from pandas.core.frame import DataFrame
from nnrecommend.dataset import Dataset
from nnrecommend.logging import get_logger
from logging import Logger
import numpy as np

class SpotifyDataset:
    """
    the dataset can be downloaded from https://aicrowd-production.s3.eu-central-1.amazonaws.com/dataset_files/challenge_25/0654d015-d4b4-4357-8040-6a846dec093d_training_set_track_features_mini.tar.gz
    """
    def __init__(self, path: str, logger: Logger=None):
        self.__path = path
        self.__logger = logger or get_logger(self)
        self.trainset = None
        self.testset = None
        self.matrix = None
        self.features = None

    STRING_ROWS = ("context_type",)

    def __load_data(self, type:str, maxsize: int):
        path = f"{self.__path}.{type}.csv"
        data = pd.read_csv(path, sep=',', nrows=maxsize)
        for row in self.STRING_ROWS:
            if row in data:
                del data[row] 
        return data

    def load(self, maxsize: int=-1) -> None:
        self.__logger.info("loading training dataset...")
        self.trainset = Dataset(self.__load_data("train", maxsize))
        iddiff = self.trainset.normalize_ids()
        self.__logger.info("loading test dataset...")
        self.testset = Dataset(self.__load_data("test", maxsize))
        self.testset.normalize_ids(iddiff)
        self.__logger.info("calculating adjacency matrix...")
        self.matrix = self.trainset.create_adjacency_matrix()