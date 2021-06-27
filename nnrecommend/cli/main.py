import click
import os
import numpy as np
import torch
import random
from typing import Container
from nnrecommend.logging import setup_log
from nnrecommend.dataset import BaseDatasetSource
from nnrecommend.dataset.movielens import MovielensLabDatasetSource, Movielens100kDatasetSource
from nnrecommend.dataset.podcasts import ItunesPodcastsDatasetSource
from nnrecommend.dataset.spotify import SpotifyDatasetSource, SpotifyRawDatasetSource
from nnrecommend.hparams import HyperParameters


DATASET_TYPES = ['movielens-lab', 'movielens', 'podcasts', 'spotify', 'spotify-raw']

class Context:

    def setup(self, verbose: bool, logoutput: str, hparams: Container[str], hparams_path: str, random_seed: int=None) -> None:
        self.logger = setup_log(verbose, logoutput)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            self.logger.warn("cuda not available, running on cpu")
        
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            random.seed(random_seed)
        self.htrials = HyperParameters.load_trials(hparams, hparams_path)
        l = len(self.htrials)
        if l > 1:
            self.logger.info(f"found {l} hparam trials...")

    def create_dataset_source(self, path, dataset_type: str) -> BaseDatasetSource:
        path = os.path.realpath(path)
        if dataset_type == "movielens-lab":
            self.logger.info("creating movielens lab dataset")
            return MovielensLabDatasetSource(path, self.logger)
        if dataset_type == "movielens":
            self.logger.info("creating movielens raw dataset")
            return Movielens100kDatasetSource(path, self.logger)
        elif dataset_type == "podcasts":
            self.logger.info("creating itunes podcasts dataset")
            return ItunesPodcastsDatasetSource(path, self.logger)
        elif dataset_type == "spotify":
            self.logger.info("creating spotify dataset")
            return SpotifyDatasetSource(path, self.logger)
        elif dataset_type == "spotify-raw":
            self.logger.info("creating spotify raw dataset")
            return SpotifyRawDatasetSource(path, self.logger)
        else:
            raise ValueError(f"unknow dataset type {dataset_type}")


@click.group()
@click.pass_context
@click.option('-v', '--verbose', type=bool, is_flag=True, help='print verbose output')
@click.option('--logoutput', type=str, help='append output to this file')
@click.option('--hparam', 'hparams', default=[], multiple=True, 
              type=str, help="hyperparam specified with name:value")
@click.option('--hparams-path', 'hparams_path', 
              type=str, help="path to json dictionary file with hyperparam values")
@click.option('--random-seed', type=int, default=42, help='random seed')
def main(ctx, verbose: bool, logoutput: str, hparams: Container[str], hparams_path: str, random_seed: int):
    """recommender system using deep learning"""
    ctx.ensure_object(Context)
    ctx.obj.setup(verbose, logoutput, hparams, hparams_path, random_seed)
    
