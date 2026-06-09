# Dual-Consistency Invariant Learning for Social Recommendation

This repository contains the PyTorch implementation of **Dual-Consistency Invariant Learning (DCIL)** for robust social recommendation.

DCIL learns noise-robust user preferences without explicitly labeling or deleting noisy social links. It builds multiple social environments with a **Social-aware View Generator**, propagates preferences with a **LightGCN-S** backbone, and jointly optimizes two paper-aligned constraints:

- **Representation Consistency (RC):** aligns the same user's embeddings across generated social-intensity views with an InfoNCE objective.
- **Prediction Consistency (PC):** minimizes cross-view prediction variance inspired by invariant risk minimization.

## Repository Layout

```text
run_DCIL.py              # Main training and evaluation entrypoint
models/base_model.py     # BaseCF embeddings and recommendation losses
models/lightgcn_s.py     # LightGCN-S social recommendation backbone
models/view_generator.py # Social-aware multi-level view generation
models/dcil.py           # DCIL with RC and PC losses
utils/rec_dataset.py     # Dataset loading and normalized social graph building
utils/evaluate.py        # Recall, Precision, and NDCG full-ranking metrics
datasets/                # Douban-Book, Yelp, and Epinions numpy data
```

## Datasets

The experiments target the three datasets used in the paper:

- `douban_book` / Douban-Book
- `yelp` / Yelp
- `epinions` / Epinions

Each dataset directory should contain:

```text
traindata.npy
testdata.npy
user_users_d.npy
```

Optional robustness files such as `attacked_user_users_0.2.npy`, `attacked_user_users_0.5.npy`, `attacked_user_users_1.0.npy`, and `attacked_user_users_2.0.npy` are used when `--social_noise_ratio` is set.

## Train

```bash
python run_DCIL.py --dataset douban_book --runid 0 --num_views 4 --lambda_rc 1.0 --lambda_pc 1.0
```

Common paper settings are encoded as defaults: embedding dimension 64, LightGCN layers 3, batch size 2048, learning rate 0.001, Gumbel temperature 0.2, and observation bias 0.5.

## Notes

The public API intentionally uses paper method names (`DCIL`, `SocialAwareViewGenerator`, `LightGCNS`, `BaseCF`) instead of the older reused SGIL/HIL names.
