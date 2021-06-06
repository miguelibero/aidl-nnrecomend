from typing import List
import click
import sys
from nnrecommend.cli.main import main, Context, DATASET_TYPES
from nnrecommend.algo import create_algorithm, ALGORITHM_TYPES
from nnrecommend.operation import RunTracker, Setup, TestResult, Tester, create_tensorboard_writer
from nnrecommend.logging import get_logger

@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default=DATASET_TYPES[0],
              type=click.Choice(DATASET_TYPES, case_sensitive=False), help="type of dataset")
@click.option('--algorithm', 'algorithm_types', default=[], multiple=True, 
              type=click.Choice(ALGORITHM_TYPES, case_sensitive=False), help="the algorithm to use to fit the data")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
@click.option('--tensorboard', 'tensorboard_dir', type=click.Path(file_okay=False, dir_okay=True), help="save tensorboard data to this path")
def fit(ctx, path: str, dataset_type: str, algorithm_types: List[str], topk: int, tensorboard_dir: str) -> None:
    """
    fit a given recommender algorithm on a dataset

    PATH: path to the dataset files
    """
    src = ctx.obj.create_dataset_source(path, dataset_type)
    logger = ctx.obj.logger or get_logger(fit)
    hparams = ctx.obj.hparams
    
    logger.info("loading dataset...")
    setup = Setup(src, logger)
    idrange = setup(hparams)

    results = []
    if isinstance(algorithm_types, str):
        algorithm_types = algorithm_types.split(",")
    if len(algorithm_types) == 0 or algorithm_types[0] == "all":
        algorithm_types = ALGORITHM_TYPES

    def log_result(result: TestResult, prefix: str=""):
        if prefix:
            prefix = f"{prefix:10s}"
        logger.info(f'{prefix}hr={result.hr:.4f} ndcg={result.ndcg:.4f} cov={result.coverage:.2f}')

    for algorithm_type in algorithm_types:
        logger.info(f"creating algorithm {algorithm_type}...")
        algo = create_algorithm(algorithm_type, hparams, idrange)

        testloader = setup.create_testloader(hparams)
        tensorboard_tag = f"{dataset_type}-{algorithm_type}"
        tb = create_tensorboard_writer(tensorboard_dir, tensorboard_tag)
        tester = Tester(algo, testloader, topk)
        tracker = RunTracker(hparams, tb)

        try:
            logger.info("fitting algorithm...")
            algo.fit(src.trainset)

            logger.info("evaluating...")
            result = tester()
            log_result(result)
            for i in range(hparams.epochs):
                tracker.track_test_result(i, result)
            results.append((algorithm_type, result))
        except Exception as e:
            logger.exception(e)
        if tb:
            tb.close()

    results.sort(key=lambda i: i[1].hr)
    logger.info("results")
    logger.info("====")
    for algorithm_type, result in results:
        log_result(result, algorithm_type)


if __name__ == "__main__":
    sys.exit(fit(obj=Context()))


