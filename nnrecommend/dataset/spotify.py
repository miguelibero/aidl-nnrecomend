from logging import Logger
import os
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from nnrecommend.hparams import HyperParameters
from nnrecommend.dataset import BaseDatasetSource, IdFinder, InteractionDataset


MIN_ITEM_INTERACTIONS = 1
MIN_USER_INTERACTIONS = 3


class SpotifyDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://aicrowd-production.s3.eu-central-1.amazonaws.com/dataset_files/challenge_25/0654d015-d4b4-4357-8040-6a846dec093d_training_set_track_features_mini.tar.gz
    with modifications in the dataset to have skipped and previous_song columns already how we want them
    """
    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    def __load_data(self, maxsize: int, load_skip: True, load_prev: True):
        nrows = maxsize if maxsize > 0 else None
        cols = ["user_id", "song_id"]
        if load_skip:
            cols.append("skipped")
        if load_prev:
            cols.append("previous_song")
        data = pd.read_csv(self.__path, sep=',', nrows=nrows, usecols=cols)

        data = np.array(data[cols], dtype=np.int64)
        labels = np.ones(data.shape[0]) # add label column with ones
        data = np.insert(data, data.shape[1], labels, axis=1)
    
        return data

    def load(self, hparams: HyperParameters) -> None:
        maxsize = hparams.max_interactions
        self._logger.info("loading data...")
        load_skip = hparams.should_have_interaction_context("skip")
        load_prev = hparams.should_have_interaction_context("previous")
        self.trainset = InteractionDataset(self.__load_data(maxsize, load_skip, load_prev))
        prev_item_col = -2 # loaded from the dataset
        self._setup(hparams, MIN_ITEM_INTERACTIONS, MIN_USER_INTERACTIONS, prev_item_col)


class SpotifyRawDatasetSource(BaseDatasetSource):
    """
    the dataset can be downloaded from https://aicrowd-production.s3.eu-central-1.amazonaws.com/dataset_files/challenge_25/0654d015-d4b4-4357-8040-6a846dec093d_training_set_track_features_mini.tar.gz
    """
    COLUMNS = ("session_id", "track_id_clean", "session_position")
    SKIP_COLUMNS = ("skip_1", "skip_2", "skip_3", "not_skipped")
    USER_COLUMN = "session_id"
    ITEM_COLUMN = "track_id_clean"
    SORT_COLUMN = 'session_position'
    MIN_REPEAT = 2
    MIN_SKIP = 3
    ITEM_INDEX_COL = "track_id"

    FILENAME = "spotify_log_mini.csv"
    ITEMS_FILENAME = "tf_mini.csv"

    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    def __load_interactions(self, maxsize: int, load_skip: False) -> np.ndarray:
        nrows = maxsize if maxsize > 0 else None
        path = self.__path
        if os.path.isdir(path):
            path = os.path.join(path, self.FILENAME)
        cols = self.COLUMNS + self.SKIP_COLUMNS if load_skip else self.COLUMNS
        data = pd.read_csv(path, sep=',', nrows=nrows, usecols=cols)
        data.sort_values(by=[self.USER_COLUMN, self.SORT_COLUMN], inplace=True, ascending=True)
        del data[self.SORT_COLUMN]

        for colname in data.select_dtypes(exclude=int):
            data[colname] = data[colname].apply(hash)
        if load_skip:
            data = self.__fix_skip(data)

        data = np.array(data, dtype=np.int64)
        labels = np.ones(data.shape[0]) # add ones as labels
        data = np.insert(data, data.shape[1], labels, axis=1)
        return data

    def __fix_skip(self, data: DataFrame) -> DataFrame:
        """
        convert skip column into
         1 * item_id if skipped >= MIN_SKIP or counts >= MIN_REPEAT
        -1 * item_id: else
        """

        def get_skip_value(series):
            c = 0
            for v in series:
                if v: break
                c += 1
            return c

        cols = list(self.SKIP_COLUMNS)
        index = data.set_index(cols).index
        skips = index.map(get_skip_value)
        data.drop(cols, axis=1, inplace=True)
        cols = [self.USER_COLUMN, self.ITEM_COLUMN]
        index = data.set_index(cols).index
        counts = index.value_counts(sort=False)
        counts = index.map(counts.to_dict())
        good = (counts >= self.MIN_REPEAT) | (skips >= self.MIN_SKIP)
        data["skip"] = (good * 2 - 1) * data[self.ITEM_COLUMN]
        return data

    def __load_items(self, mapping: np.ndarray) -> DataFrame:
        path = os.path.join(self.__path, self.ITEMS_FILENAME)
        if not os.path.isfile(path):
            self._logger.warning("could not find track features file")
            return
        data = pd.read_csv(path, index_col=False)
        mapping = IdFinder(mapping)
        data["original_item_id"] = data[self.ITEM_INDEX_COL].copy()
        data[self.ITEM_INDEX_COL] = data[self.ITEM_INDEX_COL].apply(hash).apply(mapping.find)
        data.dropna(subset=[self.ITEM_INDEX_COL], inplace=True)
        data.set_index(self.ITEM_INDEX_COL, inplace=True)
        self._logger.info(f"loaded info for {len(data)} tracks")
        return data

    def load(self, hparams: HyperParameters) -> None:
        maxsize = hparams.max_interactions
        self._logger.info("loading spotify data...")
        load_skip = hparams.should_have_interaction_context("skip")
        self.trainset = InteractionDataset(self.__load_interactions(maxsize, load_skip))
        mapping = self._setup(hparams, MIN_ITEM_INTERACTIONS, MIN_USER_INTERACTIONS)
        if hparams.recommend:
            self._logger.info("loading track features...")
            self.items = self.__load_items(mapping[1])