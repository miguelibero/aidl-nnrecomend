import click
import os
import torch
from typing import Container
from nnrecommend.logging import setup_log
from nnrecommend.dataset import BaseDatasetSource
from nnrecommend.dataset.movielens import MovielensDatasetSource
from nnrecommend.dataset.podcasts import ItunesPodcastsDatasetSource
from nnrecommend.dataset.spotify import SpotifyDatasetSource
from nnrecommend.hparams import HyperParameters


DATASET_TYPES = ['movielens', 'podcasts','spotify']

class Context:
    def __init__(self):
        if not torch.cuda.is_available():
            raise Exception("You should enable GPU runtime")
        self.device = torch.device("cuda")

    def setup(self, verbose: bool, logoutput: str, hparams: Container[str]) -> None:
        self.logger = setup_log(verbose, logoutput)
        self.hparams = HyperParameters.fromcli(hparams)

    def create_dataset_source(self, path, dataset_type: str) -> BaseDatasetSource:
        path = os.path.realpath(path)
        if dataset_type == "movielens":
            self.logger.info("creating movielens dataset")
            path = os.path.join(path, "movielens")
            return MovielensDatasetSource(path, self.logger)
        elif dataset_type == "podcasts":
            self.logger.info("creating itunes podcasts dataset")
            return ItunesPodcastsDatasetSource(path, self.logger)
        elif dataset_type == "spotify":
            self.logger.info("creating Spotify dataset")
            return SpotifyDatasetSource(path, self.logger)
        else:
            raise ValueError(f"unknow dataset type {dataset_type}")


@click.group()
@click.pass_context
@click.option('-v', '--verbose', type=bool, is_flag=True, help='print verbose output')
@click.option('--logoutput', type=str, help='append output to this file')
@click.option('--hparam', 'hparams', default=[], multiple=True, 
              type=str, help="hyperparam specified with name:value")
def main(ctx, verbose: bool, logoutput: str, hparams: Container[str]):
    """recommender system using deep learning"""
    ctx.ensure_object(Context)
    ctx.obj.setup(verbose, logoutput, hparams)
    
