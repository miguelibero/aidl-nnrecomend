from nnrecommend.logging import get_logger
import click
import matplotlib.pyplot as plt
import sklearn.decomposition as dc
import torch
import numpy as np
from nnrecommend.operation import Setup, create_tensorboard_writer
from nnrecommend.cli.main import main
from nnrecommend.fmachine import sparse_tensor_to_scipy_matrix


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
            maxids = data["maxids"]
    except:
        logger.error("failed to load model file")
        return False

    if model is None:
        logger.error("could not load model")
        return

    logger.info(f"loaded model of type {type(model)} maxids={maxids}")

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
            if i < maxids[0]:
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
@click.option('--max-interactions', type=int, default=-1, help="maximum amount of interactions (dataset will be reduced to this size if bigger)")
@click.option('--tensorboard', 'tensorboard_dir', type=click.Path(file_okay=False, dir_okay=True), help="save tensorboard data to this path")
def explore_dataset(ctx, path: str, dataset_type: str, max_interactions: int, tensorboard_dir: str) -> None:
    """
    show information about a dataset
    """

    src = ctx.obj.create_dataset_source(path, dataset_type)
    logger = ctx.obj.logger or get_logger(explore_dataset)

    setup = Setup(src, logger)
    maxids = setup(max_interactions)

    logger.info("calculating statistics...")

    nnz = src.matrix.getnnz()
    tot = np.prod(src.matrix.shape)
    logger.info(f"users-items matrix {src.matrix.shape} non-zeros {nnz} ({100*nnz/tot:.2f}%)")

    def fix_count(data):
        data = np.asarray(data).flatten()
        return data[np.nonzero(data)]

    logger.info("calculating histograms...")

    maxuser = maxids[0]
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

    tensorboard_tag = f"{dataset_type}-{src.matrix.shape[0]}"
    tb = create_tensorboard_writer(tensorboard_dir, tensorboard_tag)
    if tb:
        logger.info("saving data in tensorboard...")

        tb.add_histogram(f"users_per_item", itemcount, bins="auto")
        tb.add_histogram(f"items_per_user", usercount, bins="auto")