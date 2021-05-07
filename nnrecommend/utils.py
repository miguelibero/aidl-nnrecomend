from statistics import mean
import math
from scipy.sparse import identity
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
import torch


def getHitRatio(recommend_list, gt_item):
    if gt_item in recommend_list:
        return 1
    else:
        return 0


def getNDCG(recommend_list, gt_item):
    idx = np.where(recommend_list == gt_item)[0]
    if len(idx) > 0:
        return math.log(2)/math.log(idx+2)
    else:
        return 0


def train(model, full_dataset, optimizer, data_loader, criterion, device, topk=10, epochs=20, tb_fm=None):
    # DO EPOCHS NOW
    for epoch_i in range(epochs):
        #data_loader.dataset.negative_sampling()
        train_loss = train_one_epoch(model, optimizer, data_loader, criterion, device)
        hr, ndcg = test(model, full_dataset, device, topk=topk)

        print('\n')

        print(f'epoch {epoch_i}:')
        print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} ')
        print('\n')
        if tb_fm:
            tb_fm.add_scalar('train/loss', train_loss, epoch_i)
            tb_fm.add_scalar('eval/HR@{topk}', hr, epoch_i)
            tb_fm.add_scalar('eval/NDCG@{topk}', ndcg, epoch_i)


def train_one_epoch(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = []

    for i, (interactions) in enumerate(data_loader):
        interactions = interactions.to(device)
        targets = interactions[:,2]
        predictions = model(interactions[:,:2])
        
        loss = criterion(predictions, targets.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return mean(total_loss)


def test(model, full_dataset, device, topk=10):
    # Test the HR and NDCG for the model @topK
    model.eval()

    HR, NDCG = [], []

    for user_test in full_dataset.test_set:
        gt_item = user_test[0][1]

        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]

        HR.append(getHitRatio(recommend_list, gt_item))
        NDCG.append(getNDCG(recommend_list, gt_item))
    return mean(HR), mean(NDCG)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_adj_mx(dims, interactions):
    train_mat = sp.dok_matrix((dims, dims), dtype=np.float32)
    for x in tqdm(interactions, desc="BUILDING ADJACENCY MATRIX..."):
        train_mat[x[0], x[1]] = 1.0
        train_mat[x[1], x[0]] = 1.0

    return train_mat