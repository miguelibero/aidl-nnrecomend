from torch_geometric.nn import GCNConv # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=GCNConv#torch_geometric.nn.conv.GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix


# Linear part of the equation
class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()

        self.fc = torch.nn.Embedding(field_dims, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # self.fc(x).shape --> [batch_size, num_fields, 1]
        # torch.sum(self.fc(x), dim=1).shape --> ([batch_size, 1])
        return torch.sum(self.fc(x), dim=1) + self.bias
        #return self.fc(x).squeeze(1) + self.bias


# FM part of the equation
class FM_operation(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # field_dims == total of nodes (sum users + context)
        #self.linear = torch.nn.Linear(field_dims, 1, bias=True)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = torch.nn.Embedding(field_dims, embed_dim, sparse=False)
        self.fm = FM_operation(reduce_sum=True)

        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, interaction_pairs):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """
        out = self.linear(interaction_pairs) + self.fm(self.embedding(interaction_pairs))
        
        return out.squeeze(1)
        
    def predict(self, interactions, device):
        # return the score, inputs are numpy arrays, outputs are tensors
 
        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        output_scores = self.forward(test_interactions)
        return output_scores
    

class GraphModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, features, train_mat, attention=False):

        super().__init__()

        self.A = train_mat
        self.features = features
        if attention:
            self.GCN_module = GATConv(int(field_dims), embed_dim, heads=8, dropout=0.6)
        else:  
            self.GCN_module = GCNConv(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.GCN_module(self.features, self.A)[x]


class FactorizationMachineModel_withGCN(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim, X, A, attention=False):
        super().__init__()

        self.linear = FeaturesLinear(field_dims)
        #self.embedding = torch.nn.Embedding(field_dims, embed_dim, sparse=False)
        self.embedding = GraphModel(field_dims, embed_dim, X, A, attention=attention)
        self.fm = FM_operation(reduce_sum=True)

        #torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, interaction_pairs):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """
        out = self.linear(interaction_pairs) + self.fm(self.embedding(interaction_pairs))
        return out.squeeze(1)
        
    def predict(self, interactions, device):
        # return the score, inputs are numpy arrays, outputs are tensors
 
        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        output_scores = self.forward(test_interactions)
        return output_scores