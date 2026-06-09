# Dual-Consistency Invariant Learning for Social Recommendation

PyTorch implementation of **Dual-Consistency Invariant Learning (DCIL)** for social recommendation.

DCIL constructs multiple social-intensity views from the observed social graph, learns user and item representations with a LightGCN-S backbone, and jointly optimizes:

- **Representation Consistency (RC)** for cross-view user representation invariance.
- **Prediction Consistency (PC)** for stable recommendation scores across views.

## Repository Layout

```text
run_DCIL.py              # Training and evaluation entrypoint
models/base_model.py     # Base collaborative filtering module
models/lightgcn_s.py     # LightGCN-S backbone
models/view_generator.py # Social-aware view generator
models/dcil.py           # DCIL model with RC and PC losses
utils/rec_dataset.py     # Dataset loading and graph construction
utils/evaluate.py        # Recall, Precision, and NDCG evaluation
datasets/                # Douban-Book, Yelp, and Epinions data
```

## Datasets

The repository includes the datasets used in the experiments:

- `douban_book`
- `yelp`
- `epinions`

Each dataset directory contains:

```text
traindata.npy
testdata.npy
user_users_d.npy
```

## Train

```bash
python run_DCIL.py --dataset douban_book --runid 0 --num_views 4 --lambda_rc 1.0 --lambda_pc 1.0
```

Default settings follow the paper experiments: embedding dimension 64, 3 LightGCN layers, batch size 2048, learning rate 0.001, Gumbel temperature 0.2, and observation bias 0.5.