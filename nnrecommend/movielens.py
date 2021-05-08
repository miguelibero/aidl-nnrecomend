import numpy as np
import pandas as pd
import torch.utils.data
import scipy.sparse as sp
from tqdm import tqdm

class MovieLens100kDataset(torch.utils.data.Dataset):
    """
    MovieLens 100k Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    """

    COL_NAMES = ("user_id", 'item_id', 'label', 'timestamp')
    SEP = '\t'

    def __init__(self, dataset_path: str, num_negatives_train: int=4, num_negatives_test: int=100):

        data = pd.read_csv(f'{dataset_path}.train.rating', sep=self.SEP, header=None, names=self.COL_NAMES).to_numpy()
        test_data = pd.read_csv(f'{dataset_path}.test.rating', sep=self.SEP, header=None, names=self.COL_NAMES).to_numpy()

        # TAKE items, targets and test_items
        self.__targets = data[:, 2]
        self.__items = self.preprocess_items(data)
        self.__interactions = []

        # Save dimensions of max users and items and build training matrix
        self.__field_dims = np.max(self.__items, axis=0) + 1
        self.train_mat = self.build_adj_mx(self.__field_dims[-1], self.__items.copy())

        # Generate train interactions with 4 negative samples for each positive
        self.negative_sampling(num_negatives=num_negatives_train)
        
        # Build test set by passing as input the test item interactions
        self.test_set = self.build_test_set(self.preprocess_items(test_data),
                                            num_neg_samples_test = num_negatives_test)

    def __len__(self):
        return len(self.__interactions)

    def __getitem__(self, index):
        return self.__interactions[index]
    
    def __progress(self, elms, desc):
        if isinstance(elms, int):
            return range(elms)
        return iter(elms)
        # return tqdm(elms, desc)

    def preprocess_items(self, data):
        reindexed_items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        users, items = np.max(reindexed_items, axis=0)[:2]
        # Reindex items (we need to have [users + items] nodes with unique idx)
        reindexed_items[:, 1] = reindexed_items[:, 1] + users

        return reindexed_items

    def build_adj_mx(self, dims, interactions):
        train_mat = sp.dok_matrix((dims, dims), dtype=np.float32)
        for x in self.__progress(interactions, desc="BUILDING ADJACENCY MATRIX..."):
            train_mat[x[0], x[1]] = 1.0
            train_mat[x[1], x[0]] = 1.0

        return train_mat

    def negative_sampling(self, num_negatives=4):
        data = np.c_[(self.__items, self.__targets)].astype(int)
        max_users, max_items = self.__field_dims[:2] # number users (943), number items (2625)

        for x in self.__progress(data, desc="Performing negative sampling on test data..."):  # x are triplets (u, i , 1) 
            # Append positive interaction
            self.__interactions.append(x)
            # Copy user and maintain last position to 0. Now we will need to update neg_triplet[1] with j
            neg_triplet = np.vstack([x, ] * (num_negatives))
            neg_triplet[:, 2] = np.zeros(num_negatives)

            # Generate num_negatives negative interactions
            for idx in range(num_negatives):
                j = np.random.randint(max_users, max_items)
                # IDEA: Loop to exclude true interactions (set to 1 in adj_train) user - item
                while (x[0], j) in self.train_mat:
                    j = np.random.randint(max_users, max_items)
                neg_triplet[:, 1][idx] = j
            self.__interactions.append(neg_triplet.copy())

        self.__interactions = np.vstack(self.__interactions)
    
    def build_test_set(self, gt_test_interactions, num_neg_samples_test=99):
        max_users, max_items = self.__field_dims[:2] # number users (943), number items (2625)
        test_set = []
        for pair in self.__progress(gt_test_interactions, desc="BUILDING TEST SET..."):
            negatives = []
            for t in range(num_neg_samples_test):
                j = np.random.randint(max_users, max_items)
                while (pair[0], j) in self.train_mat or j == pair[1]:
                    j = np.random.randint(max_users, max_items)
                negatives.append(j)
            #APPEND TEST SETS FOR SINGLE USER
            single_user_test_set = np.vstack([pair, ] * (len(negatives)+1))
            single_user_test_set[:, 1][1:] = negatives
            test_set.append(single_user_test_set.copy())
        return test_set