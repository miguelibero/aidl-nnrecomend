import click
from torch.utils.data import DataLoader
from nnrecommend.fmachine import FactorizationMachineModel
from nnrecommend.trainer import Trainer
from nnrecommend.cli import main


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--dataset', 'dataset_type', default="movielens",
              type=click.Choice(['movielens'], case_sensitive=False))
@click.option('--model', 'model_type', default='linear',
              type=click.Choice(['linear', 'gcn', 'gcn-attention'], case_sensitive=False))
@click.option('--negatives-train', type=int, default=4)
@click.option('--negatives-test', type=int, default=99)
@click.option('--batch-size', type=int, default=256)
@click.option('--topk', type=int, default=10)
@click.option('--epochs', type=int, default=20)
def train(ctx, path: str, dataset_type: str, model_type: str, negatives_train: int, negatives_test: int, batch_size: int, topk: int, epochs: int):
    """
    train a model 
    """
    device = ctx.obj.device
    dataset = ctx.obj.create_dataset(path, dataset_type)

    # load data
    if not dataset:
        raise Exception("could not create dataset")
    dataset.setup(negatives_train, negatives_test)
    trainloader = DataLoader(dataset.trainset, batch_size=batch_size)
    testloader = DataLoader(dataset.testset, batch_size=negatives_test+1)
    
    # create model
    click.echo("creating model...")
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

    # train
    click.echo("training...")
    trainer = Trainer(model, trainloader, testloader, optimizer, criterion, device)

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


main.add_command(train)