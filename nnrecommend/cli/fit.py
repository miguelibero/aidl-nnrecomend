from logging import Logger
import click
import sys
import numpy as np
from typing import Container
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from nnrecommend.dataset import BaseDatasetSource, save_model
from nnrecommend.cli.main import main, Context, DATASET_TYPES
from nnrecommend.algo import create_algorithm, ALGORITHM_TYPES, DEFAULT_ALGORITHM_TYPES
from nnrecommend.operation import RunTracker, Setup, TestResult, Tester, create_tensorboard_writer
from nnrecommend.logging import get_logger
from nnrecommend.hparams import HyperParameters


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default=DATASET_TYPES[0],
              type=click.Choice(DATASET_TYPES, case_sensitive=False), help="type of dataset")
@click.option('--algorithm', 'algorithm_types', default=DEFAULT_ALGORITHM_TYPES, multiple=True, 
              type=click.Choice(ALGORITHM_TYPES, case_sensitive=False), help="the algorithm to use to fit the data")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
@click.option('--output', type=str, help="save the fitted algorythm to a file")
@click.option('--tensorboard', 'tensorboard_dir', type=click.Path(file_okay=False, dir_okay=True), help="save tensorboard data to this path")
@click.option('--tensorboard-tag', 'tensorboard_tag', type=str, help="custom tensorboard tag")
def fit(ctx, path: str, dataset_type: str, algorithm_types: Container[str], topk: int, output: str, tensorboard_dir: str, tensorboard_tag: str) -> None:
    """
    fit a given recommender algorithm on a dataset

    PATH: path to the dataset files
    """
    src = ctx.obj.create_dataset_source(path, dataset_type)
    logger = ctx.obj.logger or get_logger(fit)
    setup = Setup(src, logger)
    results = []

    for i, hparams in enumerate(ctx.obj.htrials):

        logger.info(f"using hparams {hparams}")
        
        idrange = setup(hparams)
        testloader = setup.create_testloader(hparams)

        if isinstance(algorithm_types, str):
            algorithm_types = algorithm_types.split(",")
        if len(algorithm_types) == 0 or algorithm_types[0] == "all":
            algorithm_types = ALGORITHM_TYPES

        for algorithm_type in algorithm_types:
            logger.info("====")
            tb_tag = tensorboard_tag or dataset_type
            tb_tag = hparams.get_tensorboard_tag(tb_tag, trial=i, dataset=dataset_type, algorithm=algorithm_type)
            tb = create_tensorboard_writer(tensorboard_dir, tb_tag)
            algo_output = output.format(trial=i, algorithm=algorithm_type) if output else None
            result = __fit(algorithm_type, src, testloader, hparams, idrange, logger, topk, algo_output, tb)
            results.append((tb_tag, result))

    results.sort(key=lambda i: i[1].ndcg)
    logger.info("====")
    logger.info("results")
    logger.info("----")
    for name, result in results:
        logger.info(f'{name}: {result}')


def __fit(algorithm_type: str, src: BaseDatasetSource, testloader: DataLoader, hparams: HyperParameters, idrange: np.ndarray, logger: Logger, topk: int, output: str, tb: SummaryWriter) -> TestResult:
    logger.info(f"creating algorithm {algorithm_type}...")
    algo = create_algorithm(algorithm_type, hparams, idrange)

    tester = Tester(algo, testloader, topk)
    tracker = RunTracker(hparams, tb)

    try:
        logger.info("fitting algorithm...")
        algo.fit(src.trainset)

        logger.info("evaluating...")
        result = tester()
        logger.info(f'{result}')
        for i in range(hparams.epochs):
            tracker.track_test_result(i, result)
        return result
    except Exception as e:
        logger.exception(e)
    finally:
        if output:
            logger.info("saving algorithm...")
            save_model(output, algo, src)
        if tb:
            tb.close()


if __name__ == "__main__":
    sys.exit(fit(obj=Context()))


