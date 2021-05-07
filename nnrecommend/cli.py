import click
import os
import torch
from nnrecommend.logging import setup_log
from nnrecommend.movielens import MovieLens100kDataset

class Context:
    pass


@click.group()
@click.pass_context
@click.option('-v', '--verbose', type=bool, is_flag=True, help='print verbose output')
@click.option('--logoutput', type=str, help='append output to this file')
def main(ctx, verbose: bool, logoutput: str):
    """recommender system using deep learning"""
    ctx.ensure_object(Context)
    setup_log(verbose, logoutput)
    if not torch.cuda.is_available():
        raise Exception("You should enable GPU runtime")


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--negatives-train', type=int, default=4)
@click.option('--negatives-test', type=int, default=4)
def movielens(ctx, path: str, negatives_train: int, negatives_test: int):
    """operate with the movielens dataset
    
    the dataset can be downloaded from https://drive.google.com/uc?id=1rE20sLow9sT2ULpBOOWqw2SEnpIm16OZ

    PATH path to the uncompressed dataset directory
    """
    
    path = os.path.join(path, "movielens")
    dataset = MovieLens100kDataset(path, negatives_train, negatives_test)


if __name__ == "__main__":
    sys.exit(main(obj=Context()))