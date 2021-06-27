import sys
import click
import datetime
import numpy as np
from logging import Logger
from typing import Container
from timeit import default_timer as timer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from nnrecommend.model import create_model, create_model_training, get_optimizer_lr
from nnrecommend.cli.main import main, Context, DATASET_TYPES
from nnrecommend.model import create_model, MODEL_TYPES
from nnrecommend.operation import RunTracker, Setup, TestResult, Trainer, Tester, create_tensorboard_writer
from nnrecommend.logging import get_logger
from nnrecommend.dataset import BaseDatasetSource, save_model
from nnrecommend.hparams import HyperParameters


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default=DATASET_TYPES[0],
              type=click.Choice(DATASET_TYPES, case_sensitive=False), help="type of dataset")
@click.option('--model', 'model_types', default=[MODEL_TYPES[0]], multiple=True,
              type=click.Choice(MODEL_TYPES, case_sensitive=False), help="type of model to train")
@click.option('--output', type=str, help="save the trained model to a file")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
@click.option('--trace-mem', type=bool, is_flag=True, default=False, help='trace memory consumption')
@click.option('--tensorboard', 'tensorboard_dir', type=click.Path(file_okay=False, dir_okay=True), help="save tensorboard data to this path")
@click.option('--tensorboard-tag', 'tensorboard_tag', type=str, help="custom tensorboard tag")
@click.option('--tensorboard-embedding', 'tensorboard_embedding', type=int, default=0, help="store full embedding in tensorboard every X epoch")
def train(ctx, path: str, dataset_type: str, model_types: Container[str], output: str, topk: int, trace_mem: bool, tensorboard_dir: str, tensorboard_tag: str, tensorboard_embedding: int) -> None:
    """
    train a pytorch recommender model on a given dataset

    PATH: path to the dataset files
    """

    src = ctx.obj.create_dataset_source(path, dataset_type)
    if not src:
        raise Exception("could not create dataset")
    logger = ctx.obj.logger or get_logger(train)
    device = ctx.obj.device
    setup = Setup(src, logger, trace_mem)
    results = []

    for i, hparams in enumerate(ctx.obj.htrials):
        logger.info(f"using hparams {hparams}")
        idrange = setup(hparams)

        logger.info("creating dataloaders...")
        trainloader = setup.create_trainloader(hparams)
        testloader = setup.create_testloader(hparams)

        if isinstance(model_types, str):
            model_types = model_types.split(",")
        if len(model_types) == 0 or model_types[0] == "all":
            model_types = MODEL_TYPES

        for model_type in model_types:
            logger.info("====")
            tb_tag = tensorboard_tag or dataset_type
            tb_tag = hparams.get_tensorboard_tag(tb_tag, trial=i, dataset=dataset_type, model=model_type)
            tb = create_tensorboard_writer(tensorboard_dir, tb_tag)
            model_output = output.format(trial=i, model=model_type) if output else None
            result = __train(model_type, src, hparams, idrange, trainloader, testloader, logger, tb, device, model_output, topk, tensorboard_embedding)
            results.append((tb_tag, result))

    results.sort(key=lambda i: i[1].ndcg)
    logger.info("====")
    logger.info("results")
    logger.info("----")
    for name, result in results:
        logger.info(f"{name}: {result}")


def __train(model_type: str, src: BaseDatasetSource, hparams: HyperParameters, idrange: np.ndarray, trainloader: DataLoader, testloader: DataLoader, logger: Logger, tb: SummaryWriter, device: str, output: str, topk: int, tensorboard_embedding: int) -> TestResult:
    start_time = timer()

    logger.info(f"creating model {model_type}...")

    model = create_model(model_type, src, hparams).to(device)
    criterion, optimizer, scheduler = create_model_training(model, hparams)

    tracker = RunTracker(hparams, tb, tensorboard_embedding)
    tracker.setup_embedding(idrange)

    try:
        trainer = Trainer(model, trainloader, optimizer, criterion, device)
        tester = Tester(model, testloader, topk, device)

        logger.info("evaluating...")
        model.eval()
        result = tester()
        logger.info(f'initial topk={topk} {result}')
        best_result = result

        for i in range(hparams.epochs):
            logger.info(f"training epoch {i}...")
            model.train()
            loss = trainer()
            lr = get_optimizer_lr(optimizer)
            tracker.track_model_epoch(i, model, loss, lr)
            logger.info("evaluating...")
            model.eval()
            result = tester()
            logger.info(f'{i:03}/{hparams.epochs:03} loss={loss:.4f} lr={lr:.4f} {result}')
            tracker.track_test_result(i, result)
            if scheduler:
                scheduler.step(loss)
            if result.ndcg > best_result.ndcg:
                best_result = result
        return best_result
    finally:
        tracker.track_end("run")
        if tb:
            tb.close()
        if output:
            logger.info("saving model...")
            save_model(output, model, src)
        duration = datetime.timedelta(seconds=(timer() - start_time))
        logger.info(f"elapsed time: {duration}")


if __name__ == "__main__":
    sys.exit(train(obj=Context()))