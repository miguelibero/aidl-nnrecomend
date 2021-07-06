import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from nnrecommend.dataset import load_model
import click
import torch
from typing import Container
from nnrecommend.cli.main import Context, main
from nnrecommend.logging import get_logger
from nnrecommend.operation import Finder

@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=False))
@click.option('--item', 'item_names', default=[], multiple=True, type=str, help="items that you like")
@click.option('--field', 'fields', default=[], multiple=True, type=str, help="fields in item info to check")
@click.option('--topk', type=int, default=3, help="amount of recommended items to show")
def recommend(ctx, path: str, item_names: Container[str], fields: Container[str], topk: int) -> None:
    """
    load a model and get recommendations
    """
    ctx: Context = ctx.obj
    logger = ctx.logger or get_logger(recommend)
    device = ctx.device

    logger.info("reading model file...")
    try:
        r = load_model(path)
        model: torch.nn.Module = r[0]
        idrange: np.ndarray = r[1]
        items: DataFrame = r[2]
    except:
        logger.error("failed to load model file")
        return False

    if model is None:
        logger.error("could not load model")
        return

    logger.info(f"loaded model of type {type(model)}")
    model = model.eval().to(device)

    items = items.dropna(axis=1, how='all')
    items = items.assign(rating=0)

    pd.options.display.max_colwidth = 200

    with torch.no_grad():
        pids = np.arange(idrange[0], idrange[1])
        interactions = np.zeros((len(pids), 2), dtype=np.int64)
        interactions[:, 1] = pids
        interactions = torch.from_numpy(interactions).to(device)

        finder = Finder(items, fields)
        
        for item_name in item_names:
            r = finder(item_name)
            logger.info(f"found {r}")
            logger.info("looking for recommendations...")
            interactions[:, 0] = r.id
            predictions = model(interactions)
            ratings, indices = torch.topk(predictions, topk)
            ritems = interactions[indices][:, 1]
            for id, r in zip(ritems.cpu().tolist(), ratings.cpu().tolist()):
                items.loc[id, "rating"] = r
                row = items.loc[id]
                logger.info(f"----\n{row.to_string()}")