import click
import os
import torch
from nnrecommend.logging import setup_log
from nnrecommend.dataset.movielens import MovielensDataset


class Context:
    def __init__(self):
        if not torch.cuda.is_available():
            raise Exception("You should enable GPU runtime")
        self.device = torch.device("cuda")

    def setup(self, verbose: bool, logoutput: str) -> None:
        self.logger = setup_log(verbose, logoutput)

    def create_dataset(self, path, dataset_type: str):
        if dataset_type == "movielens":
            self.logger.info("creating movielens dataset")
            path = os.path.join(path, "movielens")
            return MovielensDataset(path, self.logger)
        else:
            raise ValueError(f"unknow dataset type {dataset_type}")


@click.group()
@click.pass_context
@click.option('-v', '--verbose', type=bool, is_flag=True, help='print verbose output')
@click.option('--logoutput', type=str, help='append output to this file')
def main(ctx, verbose: bool, logoutput: str):
    """recommender system using deep learning"""
    ctx.ensure_object(Context)
    ctx.obj.setup(verbose, logoutput)
    
