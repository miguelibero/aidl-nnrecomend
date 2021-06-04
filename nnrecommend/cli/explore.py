from nnrecommend.logging import get_logger
import click
import matplotlib.pyplot as plt
import sklearn.decomposition as dc
import torch
import numpy as np
from nnrecommend.operation import Setup
from nnrecommend.cli.main import main
from nnrecommend.model import sparse_tensor_to_scipy_matrix
from matplotlib.ticker import MaxNLocator

@main.command()
@click.pass_context
@click.option('--embedding-graph', type=bool, is_flag=True, help='show an embedding graph')
@click.argument('path', type=click.Path(file_okay=True, dir_okay=False))
def explore_model(ctx, path: str, embedding_graph: bool) -> None:
    """
    show information about a model
    """
    logger = ctx.obj.logger or get_logger(explore_dataset)

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
@click.option('--type', 'dataset_type', default="movielens",
              type=click.Choice(['movielens', 'podcasts', 'spotify'], case_sensitive=False))
@click.option('--hist-bins', type=int, default=20, help="amount bins for the histograms")
def explore_dataset(ctx, path: str, dataset_type: str, hist_bins: int) -> None:
    """
    show information about a dataset
    """

    src = ctx.obj.create_dataset_source(path, dataset_type)
    logger = ctx.obj.logger or get_logger(explore_dataset)
    hparams = ctx.obj.hparams

    setup = Setup(src, logger)
    idrange = setup(hparams)

    logger.info("calculating statistics...")

    nnz = src.matrix.getnnz()
    tot = np.prod(src.matrix.shape)
    logger.info(f"users-items matrix {src.matrix.shape} non-zeros {nnz} ({100*nnz/tot:.2f}%)")

    def fix_count(data):
        data = np.asarray(data).flatten()
        return data[np.nonzero(data)]

    logger.info("calculating histograms...")

    maxuser = idrange[0]
    users = src.matrix[:maxuser, maxuser+1:]
    usercount = fix_count(users.sum(1))
    itemcount = fix_count(users.sum(0))

    def print_stats(count, name, other):
        tot = len(count)
        more2 = np.count_nonzero(count >= 2)*100/tot
        more10 = np.count_nonzero(count >= 10)*100/tot
        logger.info(f"{name} total = {tot}, 2 or more {other} = {more2:.2f}%, 10 or more {other} = {more10:.2f}%")

    print_stats(usercount, "users", "items")
    print_stats(itemcount, "items", "users")

    logger.info("generating graph...")

    def matrix_spy_graph(ax):
        ax.set_ylabel('users')
        ax.set_xlabel('items')
        ax.set_title('adjacency matrix')
        ax.spy(users, markersize=1)

    def log_histogram_graph(ax, x, log=True):
        ax.hist(x, bins=hist_bins, log=log)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def user_histogram_graph(ax):
        ax.set_title('amount of items per user')
        log_histogram_graph(ax, usercount)

    def item_histogram_graph(ax):
        ax.set_title('amount of users per item')
        log_histogram_graph(ax, itemcount)

    _, axs = plt.subplots(1, 3)
    matrix_spy_graph(axs[0])
    user_histogram_graph(axs[1])
    item_histogram_graph(axs[2])
    plt.show()