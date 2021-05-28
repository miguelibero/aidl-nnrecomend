from io import UnsupportedOperation
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import scipy.sparse as sp
import numpy as np


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
        self.matrix, _ = from_scipy_sparse_matrix(matrix)
        self.gcn = None

    def _apply(self, fn):
        super()._apply(fn)
        self.features = fn(self.features)
        self.matrix = fn(self.matrix)
        return self

    def get_embedding_weight(self):
        return self.gcn.weight

    def forward(self, x):
        return self.gcn(self.features, self.matrix)[x]


class GraphEmbedding(BaseGraphEmbedding):
    def __init__(self, embed_dim: int, matrix: sp.spmatrix, features: sp.spmatrix = None):
        super().__init__(matrix, features)
        self.gcn = GCNConv(matrix.shape[0], embed_dim)


class GraphAttentionEmbedding(BaseGraphEmbedding):
    def __init__(self, embed_dim: int, matrix: sp.spmatrix, features: sp.spmatrix = None):
        super().__init__(matrix, features)
        self.gcn = GATConv(matrix.shape[0], embed_dim, heads=8, dropout=0.6)

    def get_embedding_weight(self):
        return self.gcn.lin_r.weight.transpose(0, 1)


class BaseFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dim: int):
        super().__init__()
        self.linear = LinearFeatures(field_dim)
        self.fm = FactorizationMachineOperation(reduce_sum=True)
        self.embedding = None

    def get_embedding_weight(self):
        return self.embedding.weight

    def forward(self, interactions: torch.Tensor):
        out = self.linear(interactions) + self.fm(self.embedding(interactions))
        return out.squeeze(1)


class FactorizationMachine(BaseFactorizationMachine):

    def __init__(self, field_dim: int, embed_dim: int):
        super().__init__(field_dim)
        self.embedding = torch.nn.Embedding(field_dim, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


class GraphFactorizationMachine(BaseFactorizationMachine):

    def __init__(self, embed_dim: int, matrix: sp.spmatrix, features: sp.spmatrix = None):
        super().__init__(matrix.shape[0])
        self.embedding = GraphEmbedding(embed_dim, matrix, features)

    def get_embedding_weight(self):
        return self.embedding.get_embedding_weight()


class GraphAttentionFactorizationMachine(BaseFactorizationMachine):

    def __init__(self, embed_dim: int, matrix: sp.spmatrix, features: sp.spmatrix = None):
        super().__init__(matrix.shape[0])
        self.embedding = GraphAttentionEmbedding(embed_dim, matrix, features)

    def get_embedding_weight(self):
        return self.embedding.get_embedding_weight()


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