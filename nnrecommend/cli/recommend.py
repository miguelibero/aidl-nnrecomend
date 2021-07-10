import numpy as np
import pandas as pd
import scipy.sparse as sp
import click
import torch
import random
from typing import Container
from pandas.core.frame import DataFrame
from nnrecommend.dataset import load_model
from nnrecommend.cli.main import Context, main
from nnrecommend.logging import get_logger
from nnrecommend.operation import Finder, Recommender

@main.command()
@click.pass_context
@click.argument('path', type=click.Path(file_okay=True, dir_okay=False))
@click.option('--label', 'labels', default=[], multiple=True, type=str, help="items that you like")
@click.option('--field', 'fields', default=[], multiple=True, type=str, help="fields in item info to check")
@click.option('--topk', type=int, default=3, help="amount of recommended items to show")
@click.option('--user-items', type=int, default=0, help="enable user mode, show this amount of  user items")
def recommend(ctx, path: str, labels: Container[str], fields: Container[str], topk: int, user_items: int) -> None:
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
        matrix: sp.spmatrix = r[2]
        items: DataFrame = r[3]
    except Exception as e:
        logger.exception(e, "failed to load model file")
        return False

    if model is None:
        logger.error("could not load model")
        return

    logger.info(f"loaded idrange {idrange}")

    logger.info(f"loaded model of type {type(model)}")
    model = model.eval().to(device)

    pd.options.display.max_colwidth = 200
    if items is None:
        logger.info("no items found, generating ids dataframe...")
        ids = list(range(idrange[0]))
        items = DataFrame({'id': ids}, index=ids)

    items.index = items.index.astype(int, copy=False)
    logger.info(f"loaded {len(items)} items")

    with torch.no_grad():
        finder = Finder(items, fields)
        recommender = Recommender(idrange, items, model, device)

        ids = []
        if user_items != 0:
            assert len(labels) == 1
            uid = int(labels[0])
            assert uid > 0 and uid < idrange[0]
            ids.append(uid)
            fitems = []
            row = matrix[uid]
            logger.info(f"user {uid} interacted with {len(row)} items...")
            logger.info(f"here's {user_items} of them")
            remove_ids = []
            for key in row.keys():
                iid = key[1] - idrange[0]
                fitems.append(items.loc[iid])
                remove_ids.append(iid)
            if user_items > 0:
                fitems = random.sample(fitems, user_items)
            for item in fitems:
                logger.info(f"found item {item.name}:")
                logger.info(f"\n{item.to_string()}")
        else:
            for label in labels:
                r = finder(label)
                if r is None:
                    logger.info(f"did not find any item for '{label}'")
                    continue
                ids.append(r.id)
                item = items.loc[r.id]
                logger.info(f"found item {item.name}:")
                logger.info(f"\n{item.to_string()}")
            remove_ids = ids

        logger.info(">>>")
        logger.info("looking for recommendations...")
        for item, rating in recommender(ids, topk, remove_ids):
            logger.info(f"id:{item.name} rating:{rating:.4f}\n{item.to_string()}")


