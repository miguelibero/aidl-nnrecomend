from typing import List
import click
from surprise.prediction_algorithms.algo_base import AlgoBase
from nnrecommend.cli.main import main, Context
from nnrecommend.algo import SurpriseAlgorithm
from nnrecommend.operation import Setup, TestResult, Tester
from nnrecommend.logging import get_logger
import surprise
import sys


ALGORITHM_TYPES = ['baseline', 'normal', 'slope', 'cocluster', 'knn', 'knn-means', 'svd', 'nmf']


def create_surprise_algorithm(algorithm_type) -> AlgoBase:
    if algorithm_type == "knn":
        return surprise.prediction_algorithms.knns.KNNBasic()
    if algorithm_type == "knn-means":
        return surprise.prediction_algorithms.knns.KNNWithMeans()
    elif algorithm_type == "svd":
        return surprise.prediction_algorithms.matrix_factorization.SVD()
    elif algorithm_type == "svdpp": # takes long time
        return surprise.prediction_algorithms.matrix_factorization.SVDpp()
    elif algorithm_type == "nmf": # float division error
        return surprise.prediction_algorithms.matrix_factorization.NMF()
    elif algorithm_type == "cocluster":
        return surprise.prediction_algorithms.co_clustering.CoClustering()  
    elif algorithm_type == "slope":
        return surprise.prediction_algorithms.slope_one.SlopeOne()  
    elif algorithm_type == "normal":
        return surprise.prediction_algorithms.random_pred.NormalPredictor()  
    elif algorithm_type == "baseline":
        return surprise.prediction_algorithms.baseline_only.BaselineOnly()  
    else:
        raise Exception("could not create algorithm")


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default="movielens",
              type=click.Choice(['movielens', 'podcasts','spotify'], case_sensitive=False), help="type of dataset")
@click.option('--algorithm', 'algorithm_types', default=[], multiple=True, 
              type=click.Choice(ALGORITHM_TYPES, case_sensitive=False), help="the algorithm to use to fit the data")
@click.option('--tensorboard', 'tensorboard_dir', type=click.Path(file_okay=False, dir_okay=True), help="save tensorboard data to this path")
@click.option('--max-interactions', type=int, default=-1, help="maximum amount of interactions (dataset will be reduced to this size if bigger)")
@click.option('--negatives-train', type=int, default=4, help="amount of negative samples to generate for the trainset")
@click.option('--negatives-test', type=int, default=99, help="amount of negative samples to generate for the testset")
@click.option('--batch-size', type=int, default=256, help="batchsize of the trainset dataloader")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
def fit(ctx, path: str, dataset_type: str, algorithm_types: List[str], tensorboard_dir: str, max_interactions: int, negatives_train: int, negatives_test: int, batch_size: int, topk: int, ) -> None:
    """
    fit a given recommender algorithm on a dataset

    PATH: path to the dataset files
    """
    src = ctx.obj.create_dataset_source(path, dataset_type)
    logger = ctx.obj.logger or get_logger(fit)
    
    logger.info("loading dataset...")
    setup = Setup(src, logger)
    maxids = setup(max_interactions, negatives_train, negatives_test)

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
        algo = create_surprise_algorithm(algorithm_type)
        algo = SurpriseAlgorithm(algo, maxids[0] + 1)

        testloader = setup.create_testloader()
        algo_tb_tag = f"{dataset_type}-{algorithm_type}"
        tester = Tester(algo, testloader, src.trainset, topk, None, tensorboard_dir, algo_tb_tag)

        try:
            logger.info("fitting algorithm...")
            algo.fit(src.trainset)

            logger.info("evaluating...")
            result = tester(0)
            log_result(result)
            results.append((algorithm_type, result))
        except Exception as e:
            logger.exception(e)

    results.sort(key=lambda i: i[1].hr)

    logger.info("results")
    logger.info("====")
    for algorithm_type, result in results:
        log_result(result, algorithm_type)


if __name__ == "__main__":
    sys.exit(fit(obj=Context()))


