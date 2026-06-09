import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCF(nn.Module):
    """Base collaborative filtering module used by DCIL and LightGCN-S."""

    def __init__(self, args):
        super().__init__()
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.latent_dim = args.latent_dim
        self.l2_reg = args.l2_reg
        self.batch_size = args.batch_size
        self.device = args.device

        self.user_embedding = nn.Embedding(self.num_user, self.latent_dim)
        self.item_embedding = nn.Embedding(self.num_item, self.latent_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

    def softmax_loss_batch(self, users, pos_items, user_emb, item_emb, temperature=0.2):
        users = users.long().to(self.device)
        pos_items = pos_items.long().to(self.device)
        batch_user_emb = F.normalize(user_emb[users], p=2, dim=1)
        batch_item_emb = F.normalize(item_emb[pos_items], p=2, dim=1)

        logits = torch.matmul(batch_user_emb, batch_item_emb.t()) / temperature
        labels = torch.arange(logits.size(0), device=self.device)
        return F.cross_entropy(logits, labels, reduction="none")

    def compute_bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb):
        users = users.long().to(self.device)
        pos_items = pos_items.long().to(self.device)
        neg_items = neg_items.long().to(self.device)
        pos_scores = (user_emb[users] * item_emb[pos_items]).sum(dim=1)
        neg_scores = (user_emb[users] * item_emb[neg_items]).sum(dim=1)
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def l2_regularizer(self, tensors):
        loss = torch.tensor(0.0, device=self.device)
        for tensor in tensors:
            loss = loss + tensor.pow(2).sum()
        return self.l2_reg * loss / max(1, self.batch_size)
