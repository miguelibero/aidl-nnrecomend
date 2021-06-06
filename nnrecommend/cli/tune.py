
import click
import torch
import sys
from ray import tune as rtune
from nnrecommend.model import create_model, create_model_training
from nnrecommend.cli.main import main, Context, DATASET_TYPES
from nnrecommend.model import create_model, get_optimizer_lr, MODEL_TYPES
from nnrecommend.operation import Setup, Trainer, Tester
from nnrecommend.logging import get_logger
from nnrecommend.hparams import RayTuneConfigFile
import os


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default=DATASET_TYPES[0],
              type=click.Choice(DATASET_TYPES, case_sensitive=False), help="type of dataset")
@click.option('--model', 'model_type', default=MODEL_TYPES[0],
              type=click.Choice(MODEL_TYPES, case_sensitive=False), help="type of model to train")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
@click.option('--num-samples', type=int, default=10, help="amount of samples to tune")
@click.option('--config', 'config_path', required=True,
              type=str, help="path to json dictionary file with ray tune config values")
@click.option('--output', type=str, help="save the trained model to a file")
def tune(ctx, path: str, dataset_type: str, model_type: str, topk: int, num_samples: int, config_path: str, output: str) -> None:
    """
    train a pytorch recommender model on a given dataset

    PATH: path to the dataset files
    """
    src = ctx.obj.create_dataset_source(path, dataset_type)
    logger = ctx.obj.logger or get_logger(tune)
    device = ctx.obj.device
    config = RayTuneConfigFile.load(config_path)
    tune_metric = "ndcg"
    tune_metric_mode = "max"

    if not src:
        raise Exception("could not create dataset")

    def training_function(config):
        hparams = ctx.obj.hparams.copy(config)
        setup = Setup(src, logger)
        setup(hparams)
        model = create_model(model_type, src, hparams).to(device)
        criterion, optimizer, scheduler = create_model_training(model, hparams)
        trainloader = setup.create_trainloader(hparams)
        testloader = setup.create_testloader(hparams)
        trainer = Trainer(model, trainloader, optimizer, criterion, device)
        tester = Tester(model, testloader, topk, device)

        for i in range(hparams.epochs):
            loss = trainer()
            result = tester()
            lr = get_optimizer_lr(optimizer)
            rtune.report(mean_loss=loss, hr=result.hr, ndcg=result.ndcg, cov=result.coverage, lr=lr)
            if scheduler:
                scheduler.step(loss)

    analysis = rtune.run(
        training_function,
        config=config.generate(model_type),
        queue_trials=True,
        scheduler=rtune.schedulers.ASHAScheduler(metric=tune_metric, mode=tune_metric_mode),
        num_samples=num_samples,
        resources_per_trial={
            "cpu": 1,
            "gpu": 0.5,
        })

    config = analysis.get_best_config(metric=tune_metric, mode=tune_metric_mode)
    logger.info(f"best config {config}")

    if output:
        analysis.results_df.to_csv(output)

if __name__ == "__main__":
    sys.exit(tune(obj=Context()))