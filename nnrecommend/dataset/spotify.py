from logging import Logger
from nnrecommend.hparams import HyperParameters
import pandas as pd
from nnrecommend.dataset import BaseDatasetSource, Dataset


class SpotifyDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://aicrowd-production.s3.eu-central-1.amazonaws.com/dataset_files/challenge_25/0654d015-d4b4-4357-8040-6a846dec093d_training_set_track_features_mini.tar.gz
    """
    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    COLUMNS = ("user_id", "song_id")

    def __load_data(self, type:str, maxsize: int):
        nrows = maxsize if maxsize > 0 else None
        path = f"{self.__path}.{type}.csv"
        data = pd.read_csv(path, sep=',', nrows=nrows, usecols=self.COLUMNS)
        return data

    def load(self, hparams: HyperParameters) -> None:
        maxsize = hparams.max_interactions
        self._logger.info("loading training dataset...")
        self.trainset = Dataset(self.__load_data("train", maxsize))
        self._logger.info("normalizing ids...")
        mapping = self.trainset.normalize_ids()
        self._logger.info("calculating adjacency matrix...")
        self.matrix = self.trainset.create_adjacency_matrix()
        self._logger.info("removing low interactions...")
        ci = self.trainset.remove_low_items(self.matrix, 1)
        cu = self.trainset.remove_low_users(self.matrix, 3)
        if cu > 0 or ci > 0:
            self._logger.info(f"removed {cu} users and {ci} items")
            self._logger.info("normalizing ids again...")
            self.trainset.denormalize_ids(mapping)
            mapping = self.trainset.normalize_ids()
            self._logger.info("calculating adjacency matrix again...")
            self.matrix = self.trainset.create_adjacency_matrix()
        self._logger.info("loading test dataset...")
        self.testset = Dataset(self.__load_data("test", maxsize))
        self.testset.map_ids(mapping)