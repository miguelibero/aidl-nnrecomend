import click
import matplotlib.pyplot as plt
import sklearn.decomposition as dc
import torch
import numpy as np
from nnrecommend.cli.main import main


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=False))
def model_graph(ctx, path: str):
    """
    show graphs about a model
    """

    logger = ctx.obj.logger

    logger.info("reading model file...")
    with open(path, "rb") as fh:
        model = torch.load(fh)

    if model is None:
        logger.error("could not load model")
        return

    logger.info(f"loaded model of type {type(model)}")

    weight = model.get_embedding_weight().cpu().detach().numpy()

    logger.info(f"fitting weights of shape {weight.shape} into 2 dimensions...")
    pca = dc.SparsePCA(n_components=2)
    result = pca.fit_transform(weight)

    logger.info("generating graph...")
    plt.scatter(result[:, 0], result[:, 1])
    plt.show()


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--type', 'dataset_type', default="movielens",
              type=click.Choice(['movielens', 'podcasts'], case_sensitive=False))
@click.option('--hist-bins', type=int, default=20, help="amount bins for the histograms")
def dataset_graph(ctx, path: str, dataset_type: str, hist_bins: int):
    """
    show graphs about a dataset
    """

    logger = ctx.obj.logger

    dataset = ctx.obj.create_dataset(path, dataset_type)
    dataset.load()

    logger.info("calculating graph data...")

    def fix_count(data):
        data = np.asarray(data).flatten()
        return data[np.nonzero(data)]

    maxuser = dataset.trainset.idrange[0]
    users = dataset.matrix[:maxuser-1, maxuser:]
    usercount = fix_count(users.sum(1))
    itemcount = fix_count(users.sum(0))

    nnz = users.nnz
    tot = len(users)
    logger.info(f"users-items matrix {users.shape} non-zeros {nnz} ({100*nnz/tot:.2f}%)")

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
        hist, bins = np.histogram(x, bins=hist_bins)
        if log:
            bins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
            ax.set_xscale('log')
        ax.hist(x, bins=bins)


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
