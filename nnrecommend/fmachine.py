from io import UnsupportedOperation
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import scipy.sparse as sp
import numpy as np

class LinearFeatures(torch.nn.Module):
    """
    Linear part of the equation
    """

    def __init__(self, field_dim: int, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(field_dim, output_dim)
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
    def __init__(self, field_dim: int, embed_dim: int, features: torch.Tensor, matrix: torch.Tensor, attention: bool=False):
        super().__init__()
        self.matrix = matrix
        self.features = features
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=GCNConv#torch_geometric.nn.conv.GCNConv
        if attention:
            self.gcn = GATConv(field_dim, embed_dim, heads=8, dropout=0.6)
        else:  
            self.gcn = GCNConv(field_dim, embed_dim)

    def get_embedding_weight(self):
        if hasattr(self.gcn, "lin_r"):
            return self.gcn.lin_r.weight.transpose(0, 1)
        else:
            return self.gcn.weight

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.gcn(self.features, self.matrix)[x]


class BaseFactorizationMachineModel(torch.nn.Module):

    def __init__(self, field_dim: int):
        super().__init__()
        self.linear = LinearFeatures(field_dim)
        self.fm = FactorizationMachineOperation(reduce_sum=True)
        self.embedding = None

    def get_embedding_weight(self):
        return self.embedding.weight

    def forward(self, interactions: torch.Tensor):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """
        out = self.linear(interactions) + self.fm(self.embedding(interactions))
        return out.squeeze(1)


class FactorizationMachineModel(BaseFactorizationMachineModel):

    def __init__(self, field_dim: int, embed_dim: int):
        super().__init__(field_dim)
        self.embedding = torch.nn.Embedding(field_dim, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


class GraphFactorizationMachineModel(BaseFactorizationMachineModel):

    def __init__(self, embed_dim: int, matrix: sp.spmatrix = None, features: sp.spmatrix = None, attention: bool = False, device: str=None):
        field_dim = matrix.shape[0]
        super().__init__(field_dim)
        features = features or sp.identity(field_dim)
        features = sparse_scipy_matrix_to_tensor(features.astype(np.float32))
        indices, _ = from_scipy_sparse_matrix(matrix)
        self.embedding = GraphModel(field_dim, embed_dim, features.to(device), indices.to(device), attention)

    def get_embedding_weight(self):
        return self.embedding.get_embedding_weight()


def sparse_scipy_matrix_to_tensor(matrix: sp.spmatrix) -> torch.Tensor: 
    """ Convert a scipy sparse matrix to a torch sparse tensor."""
    matrix = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
    values = torch.from_numpy(matrix.data)
    shape = torch.Size(matrix.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sparse_tensor_to_scipy_matrix(tensor: torch.Tensor) -> sp.spmatrix:
    tensor = tensor.coalesce()
    return to_scipy_sparse_matrix(tensor.indices(), tensor.values())