import click
import os
import sys
import torch
from torch.utils.data import DataLoader
from nnrecommend.logging import setup_log
from nnrecommend.fmachine import FactorizationMachineModel
from nnrecommend.trainer import Trainer
from nnrecommend.movielens import MovielensData
import pandas as pd


class Context:
    def __init__(self):
        if not torch.cuda.is_available():
            raise Exception("You should enable GPU runtime")
        self.device = torch.device("cuda")


@click.group()
@click.pass_context
@click.option('-v', '--verbose', type=bool, is_flag=True, help='print verbose output')
@click.option('--logoutput', type=str, help='append output to this file')
def main(ctx, verbose: bool, logoutput: str):
    """recommender system using deep learning"""
    ctx.ensure_object(Context)
    setup_log(verbose, logoutput)



@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--model-type',
              type=click.Choice(['linear', 'gcn', 'gcn-attention'], case_sensitive=False))
@click.option('--negatives-train', type=int, default=4)
@click.option('--negatives-test', type=int, default=99)
@click.option('--batch-size', type=int, default=256)
@click.option('--topk', type=int, default=10)
@click.option('--epochs', type=int, default=20)
def movielens(ctx, path: str, model_type: str, negatives_train: int, negatives_test: int, batch_size: int, topk: int, epochs: int):
    """train a model with the movielens dataset
    
    the dataset can be downloaded from https://drive.google.com/uc?id=1rE20sLow9sT2ULpBOOWqw2SEnpIm16OZ

    PATH path to the uncompressed dataset directory
    """
    
    device = ctx.obj.device
    path = os.path.join(path, "movielens")
    data = MovielensData(click.echo)

    # load datasets
    data.load(path)
    data.setup(batch_size, negatives_train, negatives_test)
    
    # create model
    click.echo("creating model...")
    field_dim = data.matrix.shape[0]
    model = None
    if model_type == "gcn" or model_type == "gcn-attention":
        attention = model_type == "gcn-attention"
        model = FactorizationMachineModel(field_dim, 64, data.matrix, device, attention)
    else:
        model = FactorizationMachineModel(field_dim, 32)
    if not model:
        raise Exception("could not create model")
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # TODO: tensorboard support
    tensorboard = None

    # train
    click.echo("training...")
    trainer = Trainer(model, data.dataloader, data.testloader, optimizer, criterion, device)

    result = trainer.test(topk)
    click.echo(f'initial hr@{topk} = {result.hr:.4f} ndcg@{topk} = {result.ndcg:.4f} ')

    for i in range(epochs):
        loss = trainer()
        result = trainer.test(topk)
        click.echo(f'{i}/{epochs} loss = {loss:.4f} hr@{topk} = {result.hr:.4f} ndcg@{topk} = {result.ndcg:.4f}')
        if tensorboard:
            tensorboard.add_scalar('train/loss', loss, i)
            tensorboard.add_scalar('eval/HR@{topk}', result.hr, i)
            tensorboard.add_scalar('eval/NDCG@{topk}', result.ndcg, i)


if __name__ == "__main__":
    sys.exit(main(obj=Context()))