import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import numpy as np


class LinearFeatures(torch.nn.Module):
    """
    Linear part of the equation
    """

    def __init__(self, field_dims: int, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(field_dims, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationMachineOperation(torch.nn.Module):
    """
    FM part of the equation
    """

    def __init__(self, reduce_sum: bool=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class GraphModel(torch.nn.Module):
    def __init__(self, field_dims: int, embed_dim: int, features: torch.Tensor, matrix: torch.Tensor, attention: bool=False):
        super().__init__()
        self.matrix = matrix
        self.features = features
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=GCNConv#torch_geometric.nn.conv.GCNConv
        if attention:
            self.gcn = GATConv(field_dims, embed_dim, heads=8, dropout=0.6)
        else:  
            self.gcn = GCNConv(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.gcn(self.features, self.matrix)[x]


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dim: int, embed_dim: int, matrix: sp.dok_matrix = None, device: str=None, attention: bool = False):
        super().__init__()
        self.linear = LinearFeatures(field_dim)
        self.fm = FactorizationMachineOperation(reduce_sum=True)

        if isinstance(matrix, type(None)):
            self.embedding = torch.nn.Embedding(field_dim, embed_dim, sparse=False)
            torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        else:
            features = sparse_mx_to_torch_sparse_tensor(sp.identity(matrix.shape[0]))
            edge_idx, edge_attr = from_scipy_sparse_matrix(matrix)
            self.embedding = GraphModel(field_dim, embed_dim, features.to(device), edge_idx.to(device), attention)

    def forward(self, interaction_pairs: torch.Tensor):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """
        out = self.linear(interaction_pairs) + self.fm(self.embedding(interaction_pairs))
        return out.squeeze(1)    


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.dok_matrix) -> torch.sparse.FloatTensor: 
    """ Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)