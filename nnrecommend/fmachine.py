import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import identity
import numpy as np


# Linear part of the equation
class LinearFeatures(torch.nn.Module):

    def __init__(self, field_dims: int, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(field_dims, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # self.fc(x).shape --> [batch_size, num_fields, 1]
        # torch.sum(self.fc(x), dim=1).shape --> ([batch_size, 1])
        return torch.sum(self.fc(x), dim=1) + self.bias
        #return self.fc(x).squeeze(1) + self.bias


# FM part of the equation
class FactorizationMachineOperation(torch.nn.Module):

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
    def __init__(self, field_dims: int, embed_dim: int, features: torch.Tensor, train_mat: torch.Tensor, attention: bool=False):

        super().__init__()

        self.A = train_mat
        self.features = features
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=GCNConv#torch_geometric.nn.conv.GCNConv
        if attention:
            self.GCN_module = GATConv(field_dims, embed_dim, heads=8, dropout=0.6)
        else:  
            self.GCN_module = GCNConv(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.GCN_module(self.features, self.A)[x]


class BaseFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dim: int):
        super().__init__()
        self.linear = LinearFeatures(field_dim)
        self.fm = FactorizationMachineOperation(reduce_sum=True)
        self.embedding = None

    def forward(self, interaction_pairs: torch.Tensor):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """
        out = self.linear(interaction_pairs) + self.fm(self.embedding(interaction_pairs))
        
        return out.squeeze(1)    


class FactorizationMachineModel(BaseFactorizationMachineModel):

    def __init__(self, field_dim: int, embed_dim: int):
        super().__init__(field_dim)
        self.embedding = torch.nn.Embedding(field_dim, embed_dim, sparse=False)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


class FactorizationMachineModel_withGCN(BaseFactorizationMachineModel):

    def __init__(self, field_dim: int, embed_dim: int, train_mat, device: str, attention: bool =False):
        super().__init__(field_dim)
        X = sparse_mx_to_torch_sparse_tensor(identity(train_mat.shape[0]))
        edge_idx, edge_attr = from_scipy_sparse_matrix(train_mat)
        self.embedding = GraphModel(field_dim, embed_dim, X.to(device), edge_idx.to(device), attention=attention)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)