import click
import torch
from nnrecommend.cli.main import main, Context
from nnrecommend.fmachine import FactorizationMachine, GraphFactorizationMachine, GraphAttentionFactorizationMachine
from nnrecommend.operation import Setup, Trainer, Tester
from nnrecommend.logging import get_logger
import sys


@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=True))
@click.option('--dataset', 'dataset_type', default="movielens",
              type=click.Choice(['movielens', 'podcasts','spotify'], case_sensitive=False), help="type of dataset")
@click.option('--model', 'model_type', default='linear',
              type=click.Choice(['linear', 'gcn', 'gcn-att'], case_sensitive=False), help="type of model to train")
@click.option('--output', type=str, help="save the trained model to a file")
@click.option('--tensorboard', 'tensorboard_dir', type=click.Path(file_okay=False, dir_okay=True), help="save tensorboard data to this path")
@click.option('--max-interactions', type=int, default=-1, help="maximum amount of interactions (dataset will be reduced to this size if bigger)")
@click.option('--negatives-train', type=int, default=4, help="amount of negative samples to generate for the trainset")
@click.option('--negatives-test', type=int, default=99, help="amount of negative samples to generate for the testset")
@click.option('--batch-size', type=int, default=256, help="batchsize of the trainset dataloader")
@click.option('--topk', type=int, default=10, help="amount of elements for the test metrics")
@click.option('--epochs', type=int, default=20, help="amount of epochs to run the training")
@click.option('--embed-dim', type=int, default=64, help="size of the embedding state")
def train(ctx, path: str, dataset_type: str, model_type: str, output: str, tensorboard_dir: str, max_interactions: int, negatives_train: int, negatives_test: int, batch_size: int, topk: int, epochs: int, embed_dim: int) -> None:
    """
    train a pytorch recommender model on a given dataset

    PATH: path to the dataset files
    """
    src = ctx.obj.create_dataset_source(path, dataset_type)
    logger = ctx.obj.logger or get_logger(train)
    device = ctx.obj.device

    if not src:
        raise Exception("could not create dataset")
    setup = Setup(src, logger)
    maxids = setup(max_interactions, negatives_train, negatives_test)

    logger.info(f"creating model {model_type}...")
    model = None
    if model_type == "gcn-att":
        model = GraphAttentionFactorizationMachine(embed_dim, src.matrix)
    if model_type == "gcn":
        model = GraphFactorizationMachine(embed_dim, src.matrix)
    else:
        field_dim = src.matrix.shape[0]
        model = FactorizationMachine(field_dim, embed_dim)
    if not model:
        raise Exception("could not create model")
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    try:
        logger.info("preparing training...")
        trainloader = setup.create_trainloader(batch_size)
        testloader = setup.create_testloader()
        model_tb_tag = f"{dataset_type}-{model_type}-{embed_dim}"
        trainer = Trainer(model, trainloader, optimizer, criterion, device, tensorboard_dir, model_tb_tag)
        tester = Tester(model, testloader, src.trainset, topk, device, tensorboard_dir, model_tb_tag)

        def result_info(result):
            return f"hr={result.hr:.4f} ndcg={result.ndcg:.4f} cov={result.coverage:.2f}"

        result = tester(-1)
        logger.info(f'initial topk={topk} {result_info(result)}')

        for i in range(epochs):
            logger.info(f"training epoch {i}...")
            loss = trainer(i)
            logger.info(f"evaluating...")
            result = tester(i)
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            logger.info(f'{i:03}/{epochs:03} loss={loss:.4f} lr={lr:.4f} {result_info(result)}')

    finally:
        if output:
            logger.info("saving model...")
            data = {
                "model": model,
                "maxids": maxids
            }
            with open(output, "wb") as fh:
                torch.save(data, fh)


if __name__ == "__main__":
    sys.exit(train(obj=Context()))