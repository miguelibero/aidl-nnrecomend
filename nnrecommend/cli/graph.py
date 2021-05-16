import click
import networkx as nx
from nnrecommend.cli.main import main
import matplotlib.pyplot as plt

@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--dataset', 'dataset_type', default="movielens",
              type=click.Choice(['movielens'], case_sensitive=False))
def graph(ctx, path: str, dataset_type: str):
    """
    show graphs about a dataset
    """
    dataset = ctx.obj.create_dataset(path, dataset_type)
    dataset.load()
    click.echo("generating graph...")
    g = nx.from_scipy_sparse_matrix(dataset.matrix, create_using=nx.Graph())
    maxuser = dataset.trainset.idrange[0]
    colormap = []
    for i, node in g.nodes.items():
        isuser = i < maxuser
        node["type"] = "user" if isuser else "item"
        colormap.append("blue" if isuser else "green")

    nx.draw(g, node_color=colormap, with_labels=False)
    plt.show()
