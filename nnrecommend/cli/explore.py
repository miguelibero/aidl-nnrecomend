import itertools
from logging import Logger

from matplotlib.axis import Axis
from nnrecommend.dataset import BaseDatasetSource
import click
import torch
import matplotlib.pyplot as plt
import sklearn.decomposition as dc
from matplotlib.ticker import MaxNLocator
import numpy as np

from nnrecommend.cli.main import Context, main, DATASET_TYPES
from nnrecommend.model import sparse_tensor_to_scipy_matrix
from nnrecommend.logging import get_logger

@main.command()
@click.pass_context
@click.option('--embedding-graph', type=bool, is_flag=True, help='show an embedding graph')
@click.argument('path', type=click.Path(file_okay=True, dir_okay=False))
def explore_model(ctx, path: str, embedding_graph: bool) -> None:
    """
    show information about a model
    """
    ctx: Context = ctx.obj
    logger = ctx.logger or get_logger(explore_dataset)

    logger.info("reading model file...")
    try:
        with open(path, "rb") as fh:
            data = torch.load(fh)
            model = data["model"]
            idrange = data["idrange"]
    except:
        logger.error("failed to load model file")
        return False

    if model is None:
        logger.error("could not load model")
        return

    logger.info(f"loaded model of type {type(model)} idrange={idrange}")

    if embedding_graph:
        weight = model.get_embedding_weight().cpu().detach()
        if weight.is_sparse:
            weight = sparse_tensor_to_scipy_matrix(weight)
        else:
            weight = weight.numpy()

        logger.info(f"fitting weights of shape {weight.shape} into 2 dimensions...")
        lsa = dc.TruncatedSVD(n_components=2)
        result = lsa.fit_transform(weight)

        colors = []
        for i in range(len(result)):
            if i < idrange[0]:
                colors.append("red")
            else:
                colors.append("blue")

        logger.info("generating graph...")

        plt.scatter(result[:, 0], result[:, 1], c=colors)
        plt.show()


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--type', 'dataset_type', default=DATASET_TYPES[0],
              type=click.Choice(DATASET_TYPES, case_sensitive=False))
@click.option('--hist-bins', type=int, default=20, help="amount bins for the histograms")
@click.option('--sub', 'show_sub', type=bool, is_flag=True, help='show submatrices')
def explore_dataset(ctx, path: str, dataset_type: str, hist_bins: int, show_sub: bool) -> None:
    """
    show information about a dataset
    """
    ctx: Context = ctx.obj
    src = ctx.create_dataset_source(path, dataset_type)
    logger = ctx.logger or get_logger(explore_dataset)

    for hparams in ctx.htrials:
        hparams.pairwise_loss = False
        src.load(hparams)
        idrange = src.trainset.idrange
        __explore_dataset(src, idrange, logger, hist_bins, show_sub)


def __explore_dataset(src: BaseDatasetSource, idrange: np.ndarray, logger: Logger, hist_bins: int, show_sub: bool) -> None:

    def get_over(count, th, tot):
        return np.count_nonzero(count >= th)*100/tot if tot > 0 else 0

    def get_rangename(i):
        if i == 0: return "users"
        if i == 1: return "items"
        return f"context{i-2}"

    logger.info("calculating statistics...")

    plt.style.use("bmh")

    pairs = list(itertools.combinations(range(0, len(idrange)), 2))

    if show_sub:
        fig, axs = plt.subplots(len(pairs), 2, squeeze=False)

    i = 0
    useritem_count = None
    for (x, y) in pairs:
        xname, yname = get_rangename(x), get_rangename(y)
        submatrix = src.trainset.create_adjacency_submatrix(x, y, half=True)
        count = np.asarray(submatrix.sum(1)).flatten()
        count = count[np.nonzero(count)]
        if x == 0 and y == 1:
            useritem_count = count

        tot = len(count)
        more2 = get_over(count, 2, tot)
        more10 = get_over(count, 10, tot)
        logger.info(f"{xname}-{yname} total = {tot}, over 2 = {more2:.2f}%, over 10 = {more10:.2f}%")

        if show_sub:
            axs[i][0].set_title(f'{xname}-{yname} submatrix')
            axs[i][0].spy(submatrix, markersize=1)

            axs[i][1].set_title(f'{xname}-{yname} histogram')
            __hist(axs[i][1], count, hist_bins)

            axs[i][1].xaxis.set_major_locator(MaxNLocator(integer=True))
        i += 1

    if not show_sub:
        fig, axs = plt.subplots(1, 2)
        
        axs[0].set_title(f'adjacency matrix')
        matrix = src.trainset.create_adjacency_matrix()
        axs[0].spy(matrix, markersize=1)

        axs[1].set_title(f'user-item histogram')
        __hist(axs[1], useritem_count, hist_bins)
    
    fig.tight_layout()
    plt.show()

def __hist(ax: Axis, count: np.array, bins: int):
    def logs(v):
        return np.log(1+v).astype('int')

    n, bins, patches = ax.hist(count, bins=bins, log=True)
    maxn = logs(max(n))
    for i in range(len(patches)):
        patches[i].set_facecolor(plt.cm.viridis(logs(n[i])/maxn))