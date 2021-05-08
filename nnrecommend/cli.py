import click
import os
import torch
from torch.utils.data import DataLoader
from nnrecommend.logging import setup_log
from nnrecommend.dataset import Dataset
from nnrecommend.fmachine import FactorizationMachineModel, FactorizationMachineModel_withGCN
from nnrecommend.utils import test, train
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
    """operate with the movielens dataset
    
    the dataset can be downloaded from https://drive.google.com/uc?id=1rE20sLow9sT2ULpBOOWqw2SEnpIm16OZ

    PATH path to the uncompressed dataset directory
    """
    
    device = ctx.obj.device
    path = os.path.join(path, "movielens")

    # load datasets
    dataset = Dataset(pd.read_csv(f"{path}.train.rating", sep='\t', header=None))
    matrix = dataset.create_adjacency_matrix()
    dataset.add_negative_sampling(matrix, negatives_train)
    testset = Dataset(pd.read_csv(f"{path}.test.rating", sep='\t', header=None), dataset.iddiff)
    testset.add_negative_sampling(matrix, negatives_test)

    assert 5*99057 == len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=negatives_test+1)
    
    field_dim = matrix.shape[0]
    model = None
    if model_type == "gcn" or model_type == "gcn-attention":
        attention = model_type == "gcn-attention"
        model = FactorizationMachineModel_withGCN(field_dim, 64, matrix, device, attention)
    else:
        model = FactorizationMachineModel(field_dim, 32)
    if not model:
        raise Exception("could not create model")
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    hr, ndcg = test(model, testloader, device, topk=topk)
    click.echo(f"initial HR: {hr}")
    click.echo(f"initial NDCG: {ndcg}")
    train(model, dataloader, testloader, optimizer, criterion, device, topk, epochs)


if __name__ == "__main__":
    sys.exit(main(obj=Context()))