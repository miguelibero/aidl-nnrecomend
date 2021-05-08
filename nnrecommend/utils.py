from statistics import mean
import math
import torch
from typing import Callable, Sequence, Tuple


def get_hit_ratio(recommend_list: torch.Tensor, gt_item: torch.Tensor):
    """
    measures wheter the test item is in the topk positions of the recommendation list
    """
    if gt_item in recommend_list:
        return 1
    else:
        return 0


def get_ndcg(recommend_list: torch.Tensor, gt_item: torch.Tensor):
    """
    normalized discounted cumulative gain
    measures the ranking quality with gives information about where in the ranking is our test item
    """
    idx = (recommend_list == gt_item).nonzero(as_tuple=True)[0]
    if len(idx) > 0:
        return math.log(2)/math.log(idx[0]+2)
    else:
        return 0


def train(model, dataloader, testloader, optimizer, criterion, device, topk=10, epochs=20, tb_fm=None):
    # DO EPOCHS NOW
    for epoch_i in range(epochs):
        train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        hr, ndcg = test(model, testloader, device, topk)

        print('\n')

        print(f'epoch {epoch_i}:')
        print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} ')
        print('\n')
        if tb_fm:
            tb_fm.add_scalar('train/loss', train_loss, epoch_i)
            tb_fm.add_scalar('eval/HR@{topk}', hr, epoch_i)
            tb_fm.add_scalar('eval/NDCG@{topk}', ndcg, epoch_i)


def train_one_epoch(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: str):
    model.train()
    total_loss = []

    for interactions in dataloader:
        interactions = interactions.to(device)
        targets = interactions[:,2]
        predictions = model(interactions[:,:2])
        
        loss = criterion(predictions, targets.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return mean(total_loss)


def test(model, testloader, device, topk=10):
    # Test the HR and NDCG for the model @topK
    model.eval()
    hr, ndcg = [], []

    for user_test in testloader:
        user_test = user_test[:,:2].to(device)
        gt_item = user_test[0][1]
        predictions = model.forward(user_test)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices][:, 1]

        hr.append(get_hit_ratio(recommend_list, gt_item))
        ndcg.append(get_ndcg(recommend_list, gt_item))
    return mean(hr), mean(ndcg)