import click


@click.group()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--dataset', 'dataset_type', default="movielens",
              type=click.Choice(['movielens'], case_sensitive=False))
def plot(ctx, path: str, dataset_type: str):
    dataset = ctx.create_dataset(path, dataset_type)

