import numpy as np
import torch
import torch.nn.functional as F

from .base_model import BaseCF
from .view_generator import SocialAwareViewGenerator


class DCIL(BaseCF):
    """Dual-Consistency Invariant Learning for social recommendation."""

    def __init__(self, args, dataset):
        super().__init__(args)
        self.gcn_layer = args.gcn_layer
        self.num_views = args.num_views
        self.lambda_rc = args.lambda_rc
        self.lambda_pc = args.lambda_pc
        self.rec_temperature = args.rec_temperature
        self.rc_temperature = args.rc_temperature
        self.edge_index, self.edge_weight = self._convert_graph(dataset.uu_i_matrix)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)
        self.view_generator = SocialAwareViewGenerator(
            latent_dim=self.latent_dim,
            num_views=self.num_views,
            gumbel_temperature=args.gumbel_temperature,
            observation_bias=args.observation_bias,
        ).to(self.device)

    @staticmethod
    def _convert_graph(adj_matrix):
        coo = adj_matrix.tocoo()
        edge_index = torch.LongTensor(np.vstack([coo.row, coo.col]))
        edge_weight = torch.FloatTensor(coo.data)
        return edge_index, edge_weight

    def lightgcn_propagate(self, ego_emb, edge_weight):
        row, col = self.edge_index
        all_embs = [ego_emb]
        for _ in range(self.gcn_layer):
            messages = all_embs[-1][row] * edge_weight.unsqueeze(1)
            next_emb = torch.zeros_like(ego_emb)
            next_emb.index_add_(0, col, messages)
            all_embs.append(next_emb)
        return torch.stack(all_embs, dim=0).mean(dim=0)

    def forward(self, training=True):
        ego_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        view_edge_weights = self.view_generator(ego_emb, self.edge_index, self.edge_weight, training=training)
        view_embeddings = [self.lightgcn_propagate(ego_emb, weight) for weight in view_edge_weights]
        mean_embedding = torch.stack(view_embeddings, dim=0).mean(dim=0)
        user_emb = mean_embedding[:self.num_user]
        item_emb = mean_embedding[self.num_user:]
        return user_emb, item_emb, view_embeddings

    def compute_recommendation_loss(self, users, pos_items, user_emb, item_emb):
        return self.softmax_loss_batch(
            users, pos_items, user_emb, item_emb, temperature=self.rec_temperature
        ).mean()

    def compute_representation_consistency_loss(self, view_embeddings, users):
        if len(view_embeddings) <= 1:
            return torch.tensor(0.0, device=self.device)
        users = users.long().to(self.device)
        anchor = F.normalize(view_embeddings[0][:self.num_user][users], p=2, dim=1)
        rc_losses = []
        labels = torch.arange(anchor.size(0), device=self.device)
        for view_emb in view_embeddings[1:]:
            target = F.normalize(view_emb[:self.num_user][users], p=2, dim=1)
            logits = torch.matmul(anchor, target.t()) / self.rc_temperature
            rc_losses.append(F.cross_entropy(logits, labels))
        return torch.stack(rc_losses).mean()

    def compute_prediction_consistency_loss(self, view_embeddings, users, pos_items):
        if len(view_embeddings) <= 1:
            return torch.tensor(0.0, device=self.device)
        users = users.long().to(self.device)
        pos_items = pos_items.long().to(self.device)
        scores = []
        for view_emb in view_embeddings:
            user_view = view_emb[:self.num_user][users]
            item_view = view_emb[self.num_user:][pos_items]
            scores.append((user_view * item_view).sum(dim=1))
        stacked_scores = torch.stack(scores, dim=1)
        return stacked_scores.var(dim=1, unbiased=False).mean()

    def get_loss(self, users, pos_items):
        users = users.long().to(self.device)
        pos_items = pos_items.long().to(self.device)
        user_emb, item_emb, view_embeddings = self.forward(training=True)
        rec_loss = self.compute_recommendation_loss(users, pos_items, user_emb, item_emb)
        rc_loss = self.compute_representation_consistency_loss(view_embeddings, users)
        pc_loss = self.compute_prediction_consistency_loss(view_embeddings, users, pos_items)
        reg_loss = self.l2_regularizer([self.user_embedding.weight, self.item_embedding.weight])
        total_loss = rec_loss + self.lambda_rc * rc_loss + self.lambda_pc * pc_loss + reg_loss
        return total_loss, rec_loss, rc_loss, pc_loss

    def get_embeddings(self):
        self.eval()
        with torch.no_grad():
            user_emb, item_emb, _ = self.forward(training=False)
        return user_emb.cpu().numpy(), item_emb.cpu().numpy()
