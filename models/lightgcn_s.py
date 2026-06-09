import numpy as np
import torch

from .base_model import BaseCF


class LightGCNS(BaseCF):
    """LightGCN-S backbone over the user-item graph plus the social graph."""

    def __init__(self, args, dataset):
        super().__init__(args)
        self.gcn_layer = args.gcn_layer
        self.edge_index, self.edge_weight = self._convert_graph(dataset.uu_i_matrix)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)

    @staticmethod
    def _convert_graph(adj_matrix):
        coo = adj_matrix.tocoo()
        edge_index = torch.LongTensor(np.vstack([coo.row, coo.col]))
        edge_weight = torch.FloatTensor(coo.data)
        return edge_index, edge_weight

    def lightgcn_propagate(self, ego_emb, edge_weight=None):
        if edge_weight is None:
            edge_weight = self.edge_weight
        row, col = self.edge_index
        all_embs = [ego_emb]
        for _ in range(self.gcn_layer):
            messages = all_embs[-1][row] * edge_weight.unsqueeze(1)
            next_emb = torch.zeros_like(ego_emb)
            next_emb.index_add_(0, col, messages)
            all_embs.append(next_emb)
        return torch.stack(all_embs, dim=0).mean(dim=0)

    def forward(self, edge_weight=None):
        ego_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        final_emb = self.lightgcn_propagate(ego_emb, edge_weight=edge_weight)
        return final_emb[:self.num_user], final_emb[self.num_user:]

    def get_loss(self, users, pos_items):
        user_emb, item_emb = self.forward()
        rec_loss = self.softmax_loss_batch(users, pos_items, user_emb, item_emb).mean()
        reg_loss = self.l2_regularizer([self.user_embedding.weight, self.item_embedding.weight])
        return rec_loss + reg_loss, rec_loss

    def get_embeddings(self):
        self.eval()
        with torch.no_grad():
            user_emb, item_emb = self.forward()
        return user_emb.cpu().numpy(), item_emb.cpu().numpy()
