import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """
    basic recommender dataset class
    """
    def __init__(self):
        self.interactions = []
        self.field_dims = []
        self.test_set = []
        self.train_mat = [] # adjacency matrix