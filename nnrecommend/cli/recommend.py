import datetime
from nnrecommend.dataset import save_model
from nnrecommend.model import get_optimizer_lr
from timeit import default_timer as timer
from typing import Container
import click
import torch
from nnrecommend.cli.main import Context, DATASET_TYPES, main
from nnrecommend.logging import get_logger
from nnrecommend.operation import Finder, RunTracker, create_tensorboard_writer
from nnrecommend.recommend import Setup, Trainer, Tester, create_model, create_model_training

@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=False))
@click.option('--item', 'items', default=[], multiple=True, type=str, help="items that you like")
@click.option('--field', 'fields', default=[], multiple=True, type=str, help="fields in item info to check")
@click.option('--topk', type=int, default=10, help="amount of elements")
def recommend(ctx, path: str, items: Container[str], fields: Container[str], topk: int) -> None:
    """
    load a model and get recommendations
    """
    logger = ctx.obj.logger or get_logger(recommend)

    logger.info("reading model file...")
    try:
        with open(path, "rb") as fh:
            data = torch.load(fh)
            model = data["model"]
            idrange = data["idrange"]
            iteminfo = data["iteminfo"]
    except:
        logger.error("failed to load model file")
        return False

    if model is None:
        logger.error("could not load model")
        return

    logger.info(f"loaded model of type {type(model)}")

    finder = Finder(iteminfo, fields)
    itemids = set()
    for item in items:
        r = finder(item)
        logger.info(f"found {r}")
        itemids.add(r.id)

    # TODO: how to add a user to the model?


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default=DATASET_TYPES[0],
              type=click.Choice(DATASET_TYPES, case_sensitive=False), help="type of dataset")
@click.option('--output', type=str, help="save the trained model to a file")
@click.option('--trace-mem', type=bool, is_flag=True, default=False, help='trace memory consumption')
@click.option('--tensorboard', 'tensorboard_dir', type=click.Path(file_okay=False, dir_okay=True), help="save tensorboard data to this path")
@click.option('--tensorboard-tag', 'tensorboard_tag', type=str, help="custom tensorboard tag")
@click.option('--tensorboard-embedding', 'tensorboard_embedding', type=int, default=0, help="store full embedding in tensorboard every X epoch")
def train_recommend(ctx, path: str, dataset_type: str, output: str, trace_mem: bool, tensorboard_dir: str, tensorboard_tag: str, tensorboard_embedding: int) -> None:
    """
    train a pytorch recommender model on a given dataset

    PATH: path to the dataset files
    """

    ctx: Context = ctx.obj
    src = ctx.create_dataset_source(path, dataset_type)
    if not src:
        raise Exception("could not create dataset")
    logger = ctx.logger or get_logger(train_recommend)
    device = ctx.device
    setup = Setup(src, logger, trace_mem)

    assert len(ctx.htrials) == 1
    hparams = ctx.htrials[0]
    idrange = setup(hparams)
    items = setup.get_items()

    logger.info("creating dataloaders...")
    trainloader = setup.create_trainloader(hparams)
    testloader = setup.create_testloader(hparams)

    model_type = "recommend"
    tb_tag = tensorboard_tag or dataset_type
    tb_tag = hparams.get_tensorboard_tag(tb_tag, trial=0, dataset=dataset_type, model=model_type)
    tb = create_tensorboard_writer(tensorboard_dir, tb_tag)
    start_time = timer()

    logger.info(f"creating model...")
    model = create_model(hparams, idrange).to(device)
    criterion, optimizer, scheduler = create_model_training(model, hparams)

    tracker = RunTracker(hparams, tb, tensorboard_embedding)
    tracker.setup_embedding(idrange)

    try:
        trainer = Trainer(model, trainloader, optimizer, criterion, device)
        tester = Tester(model, testloader, device)

        logger.info("evaluating...")
        model.eval()
        result = tester()
        logger.info(f'initial {result}')
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
            if result > best_result:
                best_result = result
    except Exception as e:
        logger.exception(e)
    finally:
        tracker.track_end("run")
        if tb:
            tb.close()
        if output:
            logger.info("saving model...")
            model_output = output.format(trial=i, model=model_type)
            save_model(model_output, model, idrange, items)
        duration = datetime.timedelta(seconds=(timer() - start_time))
        logger.info(f"elapsed time: {duration}")