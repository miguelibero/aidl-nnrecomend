
import click
import torch
import sys
from ray import tune as rtune
from nnrecommend.model import create_model
from nnrecommend.cli.main import main, Context, DATASET_TYPES
from nnrecommend.model import create_model, MODEL_TYPES
from nnrecommend.operation import Setup, Trainer, Tester
from nnrecommend.logging import get_logger
from nnrecommend.hparams import HyperParameters
import os


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default=DATASET_TYPES[0],
              type=click.Choice(DATASET_TYPES, case_sensitive=False), help="type of dataset")
@click.option('--model', 'model_type', default=MODEL_TYPES[0],
              type=click.Choice(MODEL_TYPES, case_sensitive=False), help="type of model to train")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
@click.option('--output', type=str, help="save the trained model to a file")
def tune(ctx, path: str, dataset_type: str, model_type: str, topk: int, output: str) -> None:
    """
    train a pytorch recommender model on a given dataset

    PATH: path to the dataset files
    """
    src = ctx.obj.create_dataset_source(path, dataset_type)
    logger = ctx.obj.logger or get_logger(tune)
    device = ctx.obj.device

    if not src:
        raise Exception("could not create dataset")

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def training_function(config):
        hparams = ctx.obj.hparams.copy(config)
        setup = Setup(src, logger)
        setup(hparams)
        model = create_model(model_type, src, hparams).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams.learning_rate, gamma=hparams.scheduler_gamma)
        trainloader = setup.create_trainloader(hparams.batch_size)
        testloader = setup.create_testloader()
        trainer = Trainer(model, trainloader, optimizer, criterion, device)
        tester = Tester(model, testloader, topk, device)

        for i in range(hparams.epochs):
            loss = trainer()
            result = tester()
            rtune.report(mean_loss=loss, hr=result.hr, ndcg=result.ndcg, cov=result.coverage)
            scheduler.step()

    analysis = rtune.run(
        training_function,
        config=HyperParameters.tuneconfig(model_type),
        queue_trials=True,
        metric='ndcg',
        mode='max',
        resources_per_trial={
            "cpu": 1,
            "gpu": 0.5,
        })

    config = analysis.get_best_config(metric="ndcg", mode="max")
    logger.info(f"best config {config}")

    if output:
        analysis.results_df.to_csv(output)

if __name__ == "__main__":
    sys.exit(tune(obj=Context()))