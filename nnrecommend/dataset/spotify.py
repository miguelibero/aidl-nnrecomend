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
    this source loads a preprocessed dataset obtained from the big (56Gb) trainset found here
    https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge/dataset_files
    the preprocessed dataset was obtained by estracting the users with the highest amount of interactions
    """
    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    def __load_data(self, load_skip: bool=True, load_prev: bool=True):
        cols = ["user_id", "song_id"]
        if load_skip:
            cols.append("skipped")
        if load_prev:
            cols.append("previous_song")
        data = pd.read_csv(self.__path, sep=',', usecols=cols)
        for colname in data.select_dtypes(exclude=int):
            data[colname] = data[colname].apply(hash)
        return data

    def load(self, hparams: HyperParameters) -> None:
        self._logger.info("loading interactions...")
        load_skip = hparams.should_have_interaction_context("skip")
        load_prev = hparams.should_have_interaction_context("previous")
        interactions = self.__load_data(load_skip, load_prev)
        self.trainset = InteractionDataset(interactions, add_labels_col=True)
        prev_item_col = -2 # loaded from the dataset
        self._setup(hparams, MIN_ITEM_INTERACTIONS, MIN_USER_INTERACTIONS, prev_item_col)


class SpotifyMiniDatasetSource(BaseDatasetSource):
    """
    this source loads the mini dataset (Training_set_And_Track_Features_Mini) (17.2Mb) found here
    https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge/dataset_files
    it calculates the skip value automatically and also loads the items metadata into
    the self.items property
    """
    USER_COLUMN = "session_id"
    ITEM_COLUMN = "track_id_clean"
    SORT_COLUMN = 'session_position'
    COLUMNS = (USER_COLUMN, ITEM_COLUMN, SORT_COLUMN)
    SKIP_COLUMNS = ("skip_1", "skip_2", "skip_3", "not_skipped")
    SKIP_COLUMN = "skip"
    SORT_COLUMNS = (USER_COLUMN, SORT_COLUMN)
    ITEM_ID_COLUMN = "track_id"
    ORIGINAL_ITEM_ID_COLUMN = "original_track_id"

    FILENAME = "spotify_log_mini.csv"
    ITEMS_FILENAME = "tf_mini.csv"

    def __init__(self, path: str, logger: Logger=None):
        super().__init__(logger)
        self.__path = path

    def __load_interactions(self, load_skip: bool=False) -> np.ndarray:
        path = self.__path
        if os.path.isdir(path):
            path = os.path.join(path, self.FILENAME)
        cols = self.COLUMNS + self.SKIP_COLUMNS if load_skip else self.COLUMNS
        data = pd.read_csv(path, sep=',', usecols=cols)
        data.sort_values(by=list(self.SORT_COLUMNS), inplace=True, ascending=True)
        del data[self.SORT_COLUMN]
        for colname in data.select_dtypes(exclude=int):
            data[colname] = data[colname].apply(hash)
        if load_skip:
            data = self.__fix_skip(data)
        return data

    def __fix_skip(self, data: DataFrame) -> DataFrame:
        def get_value(series):
            c = 0
            for v in series:
                if v: break
                c += 1
            return c
        cols = list(self.SKIP_COLUMNS)
        index = data.set_index(cols).index
        data[self.SKIP_COLUMN] = index.map(get_value)
        data.drop(cols, axis=1, inplace=True)
        return data

    def __load_items(self, mapping: np.ndarray) -> DataFrame:
        path = os.path.join(self.__path, self.ITEMS_FILENAME)
        if not os.path.isfile(path):
            self._logger.warning("could not find track features file")
            return
        data = pd.read_csv(path, index_col=False)

        self._logger.info(f"loaded info for {len(data)} tracks")
        mapping = IdFinder(mapping)
        data[self.ORIGINAL_ITEM_ID_COLUMN] = data[self.ITEM_ID_COLUMN].copy()
        itemids = data[self.ITEM_ID_COLUMN]
        data[self.ITEM_ID_COLUMN] = itemids.apply(hash).apply(mapping.find)
        data.dropna(subset=[self.ITEM_ID_COLUMN], inplace=True)
        data.set_index(self.ITEM_ID_COLUMN, inplace=True)
        self._logger.info(f"valid info for {len(data)} tracks")
        return data

    def load(self, hparams: HyperParameters) -> None:
        self._logger.info("loading interactions...")
        load_skip = hparams.should_have_interaction_context("skip")
        interactions = self.__load_interactions(load_skip)
        self.trainset = InteractionDataset(interactions, add_labels_col=True)
        mapping = self._setup(hparams, MIN_ITEM_INTERACTIONS, MIN_USER_INTERACTIONS)
        self._logger.info("loading track features...")
        self.items = self.__load_items(mapping[1])
