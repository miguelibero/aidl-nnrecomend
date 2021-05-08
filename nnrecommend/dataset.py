import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """
    basic recommender dataset class
    """
    def __init__(self):
        self.test_set = [] # array of arrays of interactions grouped by user
        self.train_mat = [] # adjacency matrix



class TestDataset(torch.utils.data.Dataset):

    def __init__(self):