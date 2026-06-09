from collections import defaultdict
import random

import numpy as np
import scipy.sparse as sp


def _load_dict(path):
    value = np.load(path, allow_pickle=True)
    try:
        return value.item()
    except ValueError:
        return value.tolist()


class Dataset:
    """Loads social recommendation data and builds the normalized LightGCN-S graph."""

    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.num_node = self.num_user + self.num_item
        self.batch_size = args.batch_size
        self.social_noise_ratio = getattr(args, "social_noise_ratio", 0.0)
        self.load_data()
        self.training_user = []
        self.training_item = []
        for user, items in self.traindata.items():
            self.training_user.extend([user] * len(items))
            self.training_item.extend(list(items))
        self.uu_i_matrix = self.social_lightgcn_adj_matrix()

    def load_data(self):
        self.traindata = _load_dict(self.data_path + "traindata.npy")
        self.valdata = _load_dict(self.data_path + "testdata.npy")
        self.testdata = _load_dict(self.data_path + "testdata.npy")
        social_file = "user_users_d.npy"
        if self.social_noise_ratio in {0.2, 0.5, 1.0, 2.0}:
            social_file = f"attacked_user_users_{self.social_noise_ratio:.1f}.npy"
        try:
            self.user_users = _load_dict(self.data_path + social_file)
        except FileNotFoundError:
            self.user_users = defaultdict(list)

    def social_lightgcn_adj_matrix(self):
        user_np = np.asarray(self.training_user, dtype=np.int64)
        item_np = np.asarray(self.training_item, dtype=np.int64) + self.num_user
        ui_values = np.ones_like(user_np, dtype=np.float32)
        ui_adj = sp.csr_matrix((ui_values, (user_np, item_np)), shape=(self.num_node, self.num_node))
        adj = ui_adj + ui_adj.T

        social_src = []
        social_dst = []
        for user, friends in self.user_users.items():
            for friend in friends:
                if int(user) < self.num_user and int(friend) < self.num_user:
                    social_src.append(int(user))
                    social_dst.append(int(friend))
        if social_src:
            social_values = np.ones(len(social_src), dtype=np.float32)
            social_adj = sp.csr_matrix(
                (social_values, (social_src, social_dst)), shape=(self.num_node, self.num_node)
            )
            adj = adj + social_adj + social_adj.T

        rowsum = np.asarray(adj.sum(axis=1)).flatten()
        degree_inv_sqrt = np.power(rowsum + 1e-10, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
        degree_mat = sp.diags(degree_inv_sqrt)
        return degree_mat.dot(adj).dot(degree_mat).tocsr()

    def batch_sampling_softmax(self):
        indices = np.random.permutation(len(self.training_user))
        users = np.asarray(self.training_user, dtype=np.int64)
        items = np.asarray(self.training_item, dtype=np.int64)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            if len(batch_idx) > 0:
                yield users[batch_idx], items[batch_idx]

    def batch_sampling_bpr(self, num_neg=1):
        train_users = list(self.traindata.keys())
        for _ in range(max(1, len(self.training_user) // self.batch_size)):
            batch_users, batch_pos, batch_neg = [], [], []
            for _ in range(self.batch_size):
                user = random.choice(train_users)
                pos_items = list(self.traindata[user])
                if not pos_items:
                    continue
                pos_item = random.choice(pos_items)
                for _ in range(num_neg):
                    neg_item = random.randint(0, self.num_item - 1)
                    while neg_item in pos_items:
                        neg_item = random.randint(0, self.num_item - 1)
                    batch_users.append(user)
                    batch_pos.append(pos_item)
                    batch_neg.append(neg_item)
            if batch_users:
                yield np.asarray(batch_users), np.asarray(batch_pos), np.asarray(batch_neg)
