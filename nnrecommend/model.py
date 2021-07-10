from typing import Callable
import scipy.sparse as sp
import numpy as np
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from nnrecommend.hparams import HyperParameters

class LinearFeatures(torch.nn.Module):

    def __init__(self, field_dim: int, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(field_dim, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x: torch.Tensor):
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationMachineOperation(torch.nn.Module):

    def __init__(self, reduce_sum: bool=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class BaseGraphEmbedding(torch.nn.Module):
    def __init__(self, matrix: sp.spmatrix, features: sp.spmatrix = None):
        super().__init__()
        features = features or sp.identity(matrix.shape[0])
        self.features = sparse_scipy_matrix_to_tensor(features.astype(np.float32))
        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(matrix)
        self.edge_weight = self.edge_weight.float()
        self.gcn = None

    def _apply(self, fn):
        super()._apply(fn)
        self.features = fn(self.features)
        self.edge_index = fn(self.edge_index)
        self.edge_weight = fn(self.edge_weight)
        return self

    def get_embedding_weight(self):
        return self.gcn.weight

    def forward(self, x):
        return self.gcn(self.features, self.edge_index)[x]


class GraphEmbedding(BaseGraphEmbedding):
    def __init__(self, embed_dim: int, matrix: sp.spmatrix, features: sp.spmatrix = None):
        super().__init__(matrix, features)
        self.gcn = GCNConv(matrix.shape[0], embed_dim)


class GraphAttentionEmbedding(BaseGraphEmbedding):
    def __init__(self, embed_dim: int, matrix: sp.spmatrix, heads: int=8, dropout: float=0.6, features: sp.spmatrix = None):
        super().__init__(matrix, features)
        self.gcn = GATConv(matrix.shape[0], embed_dim, heads=heads, dropout=dropout)

    def get_embedding_weight(self):
        return self.gcn.lin_r.weight.transpose(0, 1)


class BaseFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dim: int, dropout: int=0.0):
        super().__init__()
        self.linear = LinearFeatures(field_dim)
        #self.linear = torch.nn.Linear(field_dim, 1, bias=True)
        self.fm = FactorizationMachineOperation(reduce_sum=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = None

    def get_embedding_weight(self):
        return self.embedding.weight

    def forward(self, interactions: torch.Tensor):
        out = self.embedding(interactions)
        out = self.dropout(out)
        out = self.linear(interactions) + self.fm(out)
        return out.squeeze(1)


class FactorizationMachine(BaseFactorizationMachine):

    def __init__(self, field_dim: int, embed_dim: int, dropout: int=0.0):
        super().__init__(field_dim, dropout)
        self.embedding = torch.nn.Embedding(field_dim, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        #torch.nn.init.normal_(self.embedding.weight, std=0.01)
        #torch.nn.init.constant_(self.embedding.weight, 0.0)


class GraphFactorizationMachine(BaseFactorizationMachine):

    def __init__(self, embed_dim: int, matrix: sp.spmatrix, features: sp.spmatrix = None, dropout: int=0.0):
        super().__init__(matrix.shape[0], dropout)
        self.embedding = GraphEmbedding(embed_dim, matrix, features)

    def get_embedding_weight(self):
        return self.embedding.get_embedding_weight()


class GraphAttentionFactorizationMachine(BaseFactorizationMachine):

    def __init__(self, embed_dim: int, matrix: sp.spmatrix, heads: int=8, dropout: float=0.6, features: sp.spmatrix = None):
        super().__init__(matrix.shape[0])
        self.embedding = GraphAttentionEmbedding(embed_dim, matrix, heads, dropout, features)

    def get_embedding_weight(self):
        return self.embedding.get_embedding_weight()


class BPRLoss:
    """
    Bayesian Personalized Ranking loss
    https://arxiv.org/pdf/1205.2618.pdf
    """
    def __call__(self, positive_predictions: torch.Tensor, negative_predictions: torch.Tensor):
        return -(positive_predictions - negative_predictions).sigmoid().log().mean()


def sparse_scipy_matrix_to_tensor(matrix: sp.spmatrix) -> torch.Tensor: 
    """convert a scipy sparse matrix to a torch sparse tensor"""
    matrix = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
    values = torch.from_numpy(matrix.data)
    shape = torch.Size(matrix.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sparse_tensor_to_scipy_matrix(tensor: torch.Tensor) -> sp.spmatrix:
    """convert a torch sparse tensor into a scipy sparse matrix."""
    tensor = tensor.coalesce()
    return to_scipy_sparse_matrix(tensor.indices(), tensor.values())


MODEL_TYPES = ['fm-linear', 'fm-gcn', 'fm-gcn-att']


def create_model(model_type: str, hparams: HyperParameters, idrange: np.ndarray, matrix_source: Callable[[HyperParameters], sp.spmatrix]) -> torch.nn.Module:
    if model_type == "fm-gcn-att":
        matrix = matrix_source(hparams)
        return GraphAttentionFactorizationMachine(hparams.embed_dim, matrix, hparams.graph_attention_heads, hparams.embed_dropout)
    if model_type == "fm-gcn":
        matrix = matrix_source(hparams)
        return GraphFactorizationMachine(hparams.embed_dim, matrix, dropout=hparams.embed_dropout)
    elif model_type == "fm-linear" or not model_type:
        return FactorizationMachine(idrange[-1], hparams.embed_dim, dropout=hparams.embed_dropout)
    raise Exception("could not create model")


def create_model_training(model: torch.nn.Module, hparams: HyperParameters):
    if hparams.pairwise_loss:
        criterion = BPRLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams.learning_rate)
    if hparams.lr_scheduler_factor < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            patience=hparams.lr_scheduler_patience,
            factor=hparams.lr_scheduler_factor,
            threshold=hparams.lr_scheduler_threshold)
    else:
        scheduler = None
    return criterion, optimizer, scheduler


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0