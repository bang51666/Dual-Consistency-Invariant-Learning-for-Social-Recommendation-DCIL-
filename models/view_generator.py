import torch
import torch.nn as nn


class SocialAwareViewGenerator(nn.Module):
    """Semantic multi-level social intensity view generator from the DCIL paper."""

    def __init__(self, latent_dim, num_views=4, gumbel_temperature=0.2, observation_bias=0.5):
        super().__init__()
        if num_views < 1:
            raise ValueError("num_views must be positive")
        self.num_views = num_views
        self.gumbel_temperature = gumbel_temperature
        self.observation_bias = observation_bias
        self.edge_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )
        self.social_intensity = nn.Parameter(self._initial_social_intensity(num_views))
        self.view_bias = nn.Parameter(torch.full((num_views,), observation_bias))

    @staticmethod
    def _initial_social_intensity(num_views):
        if num_views == 1:
            return torch.tensor([0.0], dtype=torch.float32)
        return torch.linspace(1.5, -1.5, steps=num_views, dtype=torch.float32)

    def forward(self, ego_emb, edge_index, edge_weight, training=True):
        src_emb = ego_emb[edge_index[0]]
        dst_emb = ego_emb[edge_index[1]]
        edge_features = torch.cat([src_emb, dst_emb], dim=1)
        base_logits = self.edge_mlp(edge_features).squeeze(-1)

        masked_weights = []
        for view_idx in range(self.num_views):
            view_logits = base_logits + self.social_intensity[view_idx] * edge_weight + self.view_bias[view_idx]
            if training:
                eps = torch.rand_like(view_logits).clamp_(1e-8, 1 - 1e-8)
                gumbel_noise = torch.log(eps) - torch.log(1 - eps)
                view_logits = (view_logits + gumbel_noise) / self.gumbel_temperature
            retention_prob = torch.sigmoid(view_logits)
            masked_weights.append(edge_weight * retention_prob)
        return masked_weights
