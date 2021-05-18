import click
import torch
from torch.utils.data import DataLoader
from nnrecommend.cli.main import main
from nnrecommend.fmachine import FactorizationMachineModel
from nnrecommend.trainer import Trainer
from nnrecommend.logging import get_logger


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--dataset', 'dataset_type', default="movielens",
              type=click.Choice(['movielens'], case_sensitive=False), help="type of dataset")
@click.option('--model', 'model_type', default='linear',
              type=click.Choice(['linear', 'gcn', 'gcn-attention'], case_sensitive=False), help="type of model to train")
@click.option('--output', type=str, help="save the trained model to a file")
@click.option('--negatives-train', type=int, default=4, help="amount of negative samples to generate for the trainset")
@click.option('--negatives-test', type=int, default=99, help="amount of negative samples to generate for the testset")
@click.option('--batch-size', type=int, default=256, help="batchsize of the trainset dataloader")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
@click.option('--epochs', type=int, default=20, help="amount of epochs to run the training")
def train(ctx, path: str, dataset_type: str, model_type: str, output: str, negatives_train: int, negatives_test: int, batch_size: int, topk: int, epochs: int):
    """
    train a model 

    PATH: path to the dataset files
    """
    device = ctx.obj.device
    dataset = ctx.obj.create_dataset(path, dataset_type)
    logger = ctx.obj.logger or get_logger(train)

    # load data
    if not dataset:
        raise Exception("could not create dataset")
    dataset.setup(negatives_train, negatives_test)
    trainloader = DataLoader(dataset.trainset, batch_size=batch_size)
    testloader = DataLoader(dataset.testset, batch_size=negatives_test+1)
    
    # create model
    logger.info("creating model...")
    field_dim = dataset.matrix.shape[0]
    model = None
    if model_type == "gcn" or model_type == "gcn-attention":
        attention = model_type == "gcn-attention"
        model = FactorizationMachineModel(field_dim, 64, dataset.matrix, device, attention)
    else:
        model = FactorizationMachineModel(field_dim, 32)
    if not model:
        raise Exception("could not create model")
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # TODO: tensorboard support
    tensorboard = None

    try:
        # train
        logger.info("training...")
        trainer = Trainer(model, trainloader, testloader, optimizer, criterion, device)

        result = trainer.test(topk)
        logger.info(f'initial hr@{topk} = {result.hr:.4f} ndcg@{topk} = {result.ndcg:.4f} ')

        for i in range(epochs):
            loss = trainer()
            result = trainer.test(topk)
            logger.info(f'{i}/{epochs} loss = {loss:.4f} hr@{topk} = {result.hr:.4f} ndcg@{topk} = {result.ndcg:.4f}')
            if tensorboard:
                tensorboard.add_scalar('train/loss', loss, i)
                tensorboard.add_scalar('eval/HR@{topk}', result.hr, i)
                tensorboard.add_scalar('eval/NDCG@{topk}', result.ndcg, i)
    finally:
        if output:
            logger.info("saving model...")
            with open(output, "wb") as fh:
                torch.save(model, fh)