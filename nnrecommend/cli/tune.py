
import math
import sys
import click
from ray import tune as rtune
from nnrecommend.model import create_model, create_model_training, get_optimizer_lr, MODEL_TYPES
from nnrecommend.cli.main import main, Context, DATASET_TYPES
from nnrecommend.operation import Setup, Trainer, Tester
from nnrecommend.logging import get_logger
from nnrecommend.hparams import RayTuneConfigFile


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default=DATASET_TYPES[0],
              type=click.Choice(DATASET_TYPES, case_sensitive=False), help="type of dataset")
@click.option('--model', 'model_type', default=MODEL_TYPES[0],
              type=click.Choice(MODEL_TYPES, case_sensitive=False), help="type of model to train")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
@click.option('--num-samples', type=int, default=1, help="amount of samples to tune")
@click.option('--trial-cpu', type=float, default=1, help="amount of cpus per trial")
@click.option('--trial-gpu', type=float, default=0.5, help="amount of gpus per trial")
@click.option('--config', 'config_path', required=True,
              type=str, help="path to json dictionary file with ray tune config values")
@click.option('--output', type=str, help="save the trained model to a file")
def tune(ctx, path: str, dataset_type: str, model_type: str, topk: int, num_samples: int, trial_cpu: float, trial_gpu: float, config_path: str, output: str) -> None:
    """
    hyperparameter tuning of a pytorch recommender model on a given dataset

    PATH: path to the dataset files
    """
    ctx: Context = ctx.obj
    src = ctx.create_dataset_source(path, dataset_type)
    logger = ctx.logger or get_logger(tune)
    device = ctx.device
    config = RayTuneConfigFile.load(config_path)
    tune_metric = "ndcg"
    tune_metric_mode = "max"

    if not src:
        raise Exception("could not create dataset")

    for i, hparams in enumerate(ctx.htrials):

        def training_function(config):
            thparams = hparams.copy(config)
            setup = Setup(src, logger)
            idrange = setup(thparams)
            trainloader = setup.create_trainloader(thparams)
            testloader = setup.create_testloader(thparams)
            matrix_src = setup.create_adjacency_matrix
            model = create_model(model_type, thparams, idrange, matrix_src).to(device)
            criterion, optimizer, scheduler = create_model_training(model, thparams)
            trainer = Trainer(model, trainloader, optimizer, criterion, device)
            tester = Tester(model, testloader, topk, device)

            for _ in range(thparams.epochs):
                loss = trainer()
                result = tester()
                lr = get_optimizer_lr(optimizer)
                rtune.report(mean_loss=loss, hr=result.hr, ndcg=result.ndcg, cov=result.coverage, lr=lr)
                if math.isnan(loss):
                    return False
                if scheduler:
                    scheduler.step(loss)

        analysis = rtune.run(
            training_function,
            config=config.generate(model_type),
            queue_trials=True,
            scheduler=rtune.schedulers.ASHAScheduler(metric=tune_metric, mode=tune_metric_mode),
            num_samples=num_samples,
            resources_per_trial={
                "cpu": trial_cpu,
                "gpu": trial_gpu,
            })

        config = analysis.get_best_config(metric=tune_metric, mode=tune_metric_mode)
        logger.info(f"best config {config}")

        if output:
            analysis.results_df.to_csv(output.format(trial=i))

if __name__ == "__main__":
    sys.exit(tune(obj=Context()))