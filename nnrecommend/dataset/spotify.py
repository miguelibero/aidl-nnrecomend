from logging import Logger
import pandas as pd
from nnrecommend.dataset import BaseDatasetSource, Dataset


class SpotifyDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://aicrowd-production.s3.eu-central-1.amazonaws.com/dataset_files/challenge_25/0654d015-d4b4-4357-8040-6a846dec093d_training_set_track_features_mini.tar.gz
    """
    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    STRING_ROWS = ("context_type",)

    def __load_data(self, type:str, maxsize: int):
        nrows = maxsize if maxsize > 0 else None
        path = f"{self.__path}.{type}.csv"
        data = pd.read_csv(path, sep=',', nrows=nrows)
        for row in self.STRING_ROWS:
            if row in data:
                del data[row]
        return data

    def load(self, maxsize: int=-1) -> None:
        self._logger.info("loading training dataset...")
        self.trainset = Dataset(self.__load_data("train", maxsize))
        self._logger.info("normalizing ids...")
        mapping = self.trainset.normalize_ids()
        self._logger.info("calculating adjacency matrix...")
        self.matrix = self.trainset.create_adjacency_matrix()
        self._logger.info("removing low interactions...")
        c = self.trainset.remove_low(self.matrix, 1)
        if c > 0:
            self._logger.info(f"removed {c} interactions")
            self._logger.info("normalizing ids again...")
            self.trainset.denormalize_ids(mapping)
            mapping = self.trainset.normalize_ids()
            self._logger.info("calculating adjacency matrix again...")
            self.matrix = self.trainset.create_adjacency_matrix()
        self._logger.info("loading test dataset...")
        self.testset = Dataset(self.__load_data("test", maxsize))
        self.testset.normalize_ids(mapping)
        if c > 0:
            self._logger.info(f"removed {c} test interactions")