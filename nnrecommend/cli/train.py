
import click
import torch
import sys

from nnrecommend.model import create_model
from nnrecommend.cli.main import main, Context
from nnrecommend.model import create_model
from nnrecommend.operation import RunTracker, Setup, TestResult, Trainer, Tester, create_tensorboard_writer
from nnrecommend.logging import get_logger


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default="movielens",
              type=click.Choice(['movielens', 'podcasts','spotify'], case_sensitive=False), help="type of dataset")
@click.option('--model', 'model_type', default='linear',
              type=click.Choice(['linear', 'gcn', 'gcn-att'], case_sensitive=False), help="type of model to train")
@click.option('--output', type=str, help="save the trained model to a file")
@click.option('--tensorboard', 'tensorboard_dir', type=click.Path(file_okay=False, dir_okay=True), help="save tensorboard data to this path")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
def train(ctx, path: str, dataset_type: str, model_type: str, output: str, tensorboard_dir: str, topk: int, ) -> None:
    """
    train a pytorch recommender model on a given dataset

    PATH: path to the dataset files
    """
    src = ctx.obj.create_dataset_source(path, dataset_type)
    logger = ctx.obj.logger or get_logger(train)
    device = ctx.obj.device
    hparams = ctx.obj.hparams
    tb = create_tensorboard_writer(tensorboard_dir, f"{dataset_type}-{model_type}")

    if not src:
        raise Exception("could not create dataset")
    setup = Setup(src, logger)
    idrange = setup(hparams)

    logger.info(f"creating model {model_type}...")

    model = create_model(model_type, src, hparams).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams.learning_rate, gamma=hparams.scheduler_gamma)

    tracker = RunTracker(hparams, tb)
    tracker.setup_embedding(src.trainset.idrange)

    try:
        logger.info("preparing training...")
        trainloader = setup.create_trainloader(hparams.batch_size)
        testloader = setup.create_testloader()
        trainer = Trainer(model, trainloader, optimizer, criterion, device)
        tester = Tester(model, testloader, topk, device)

        def result_info(result):
            return f"hr={result.hr:.4f} ndcg={result.ndcg:.4f} cov={result.coverage:.2f}"

        result = tester()
        logger.info(f'initial topk={topk} {result_info(result)}')

        for i in range(hparams.epochs):
            logger.info(f"training epoch {i}...")
            loss = trainer()
            tracker.track_model_epoch(i, model, loss)
            logger.info(f"evaluating...")
            result = tester()
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            logger.info(f'{i:03}/{hparams.epochs:03} loss={loss:.4f} lr={lr:.4f} {result_info(result)}')
            tracker.track_test_result(i, result)

    finally:
        tracker.track_end()
        if tb:
            tb.close()
        if output:
            logger.info("saving model...")
            data = {
                "model": model,
                "idrange": idrange
            }
            with open(output, "wb") as fh:
                torch.save(data, fh)


if __name__ == "__main__":
    sys.exit(train(obj=Context()))