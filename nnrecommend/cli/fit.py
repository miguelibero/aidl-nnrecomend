import click
import sys
from typing import Container
from nnrecommend.cli.main import main, Context, DATASET_TYPES
from nnrecommend.algo import create_algorithm, ALGORITHM_TYPES, DEFAULT_ALGORITHM_TYPES
from nnrecommend.operation import RunTracker, Setup, Tester, create_tensorboard_writer
from nnrecommend.logging import get_logger


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default=DATASET_TYPES[0],
              type=click.Choice(DATASET_TYPES, case_sensitive=False), help="type of dataset")
@click.option('--algorithm', 'algorithm_types', default=DEFAULT_ALGORITHM_TYPES, multiple=True, 
              type=click.Choice(ALGORITHM_TYPES, case_sensitive=False), help="the algorithm to use to fit the data")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
@click.option('--tensorboard', 'tensorboard_dir', type=click.Path(file_okay=False, dir_okay=True), help="save tensorboard data to this path")
@click.option('--tensorboard-tag', 'tensorboard_tag', type=str, help="custom tensorboard tag")
def fit(ctx, path: str, dataset_type: str, algorithm_types: Container[str], topk: int, tensorboard_dir: str, tensorboard_tag: str) -> None:
    """
    fit a given recommender algorithm on a dataset

    PATH: path to the dataset files
    """
    ctx: Context = ctx.obj
    src = ctx.create_dataset_source(path, dataset_type)
    logger = ctx.logger or get_logger(fit)
    setup = Setup(src, logger)
    results = []

    for i, hparams in enumerate(ctx.htrials):
        hparams.pairwise_loss = False

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
                results.append((tb_tag, result))
            except Exception as e:
                logger.exception(e)
            finally:
                if tb:
                    tb.close()

    results.sort(key=lambda i: i[1].ndcg)
    logger.info("====")
    logger.info("results")
    logger.info("----")
    for name, result in results:
        logger.info(f'{name}: {result}')


if __name__ == "__main__":
    sys.exit(fit(obj=Context()))


