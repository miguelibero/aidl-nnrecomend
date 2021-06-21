
from nnrecommend.dataset import save_model
import click
import torch
import sys

from nnrecommend.model import create_model, create_model_training, get_optimizer_lr
from nnrecommend.cli.main import main, Context, DATASET_TYPES
from nnrecommend.model import create_model, MODEL_TYPES
from nnrecommend.operation import RunTracker, Setup, Trainer, Tester, create_tensorboard_writer
from nnrecommend.logging import get_logger


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default=DATASET_TYPES[0],
              type=click.Choice(DATASET_TYPES, case_sensitive=False), help="type of dataset")
@click.option('--model', 'model_type', default=MODEL_TYPES[0],
              type=click.Choice(MODEL_TYPES, case_sensitive=False), help="type of model to train")
@click.option('--output', type=str, help="save the trained model to a file")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
@click.option('--tensorboard', 'tensorboard_dir', type=click.Path(file_okay=False, dir_okay=True), help="save tensorboard data to this path")
@click.option('--tensorboard-tag', 'tensorboard_tag', type=str, help="custom tensorboard tag")
def train(ctx, path: str, dataset_type: str, model_type: str, output: str, topk: int, tensorboard_dir: str, tensorboard_tag: str) -> None:
    """
    train a pytorch recommender model on a given dataset

    PATH: path to the dataset files
    """
    src = ctx.obj.create_dataset_source(path, dataset_type)
    logger = ctx.obj.logger or get_logger(train)
    device = ctx.obj.device
    hparams = ctx.obj.hparams

    logger.info(f"using hparams {hparams}")
    tensorboard_tag = tensorboard_tag or f"{dataset_type}-{model_type}"
    tb = create_tensorboard_writer(tensorboard_dir, tensorboard_tag)

    if not src:
        raise Exception("could not create dataset")
    setup = Setup(src, logger)
    idrange = setup(hparams)

    logger.info(f"creating model {model_type}...")

    model = create_model(model_type, src, hparams).to(device)
    criterion, optimizer, scheduler = create_model_training(model, hparams)

    tracker = RunTracker(hparams, tb)
    tracker.setup_embedding(src.trainset.idrange)

    try:
        logger.info("preparing training...")
        trainloader = setup.create_trainloader(hparams)
        testloader = setup.create_testloader(hparams)
        trainer = Trainer(model, trainloader, optimizer, criterion, device)
        tester = Tester(model, testloader, topk, device)

        def result_info(result):
            return f"hr={result.hr:.4f} ndcg={result.ndcg:.4f} cov={result.coverage:.2f}"

        model.eval()
        result = tester()
        logger.info(f'initial topk={topk} {result_info(result)}')

        for i in range(hparams.epochs):
            logger.info(f"training epoch {i}...")
            model.train()
            loss = trainer()
            lr = get_optimizer_lr(optimizer)
            tracker.track_model_epoch(i, model, loss, lr)
            logger.info(f"evaluating...")
            model.eval()
            result = tester()
            logger.info(f'{i:03}/{hparams.epochs:03} loss={loss:.4f} lr={lr:.4f} {result_info(result)}')
            tracker.track_test_result(i, result)
            if scheduler:
                scheduler.step(loss)

    finally:
        tracker.track_end("run")
        if tb:
            tb.close()
        if output:
            logger.info("saving model...")
            save_model(output, model, src)


if __name__ == "__main__":
    sys.exit(train(obj=Context()))