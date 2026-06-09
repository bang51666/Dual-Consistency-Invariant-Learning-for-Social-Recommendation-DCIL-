import argparse
import os
from pathlib import Path
from time import time

import numpy as np
import torch
from tqdm import tqdm

from models import DCIL
from utils import Dataset, Logger, ranking_metrics


DATASET_CONFIG = {
    "douban_book": {"num_user": 13024, "num_item": 22347},
    "yelp": {"num_user": 19539, "num_item": 22228},
    "epinions": {"num_user": 18202, "num_item": 47449},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCIL for social recommendation")
    parser.add_argument("--dataset", type=str, default="douban_book", choices=sorted(DATASET_CONFIG))
    parser.add_argument("--runid", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--early_stops", type=int, default=10)
    parser.add_argument("--social_noise_ratio", type=float, default=0.0)
    parser.add_argument("--gcn_layer", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--l2_reg", type=float, default=1e-4)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--lambda_rc", type=float, default=1.0)
    parser.add_argument("--lambda_pc", type=float, default=1.0)
    parser.add_argument("--rec_temperature", type=float, default=0.2)
    parser.add_argument("--rc_temperature", type=float, default=0.2)
    parser.add_argument("--gumbel_temperature", type=float, default=0.2)
    parser.add_argument("--observation_bias", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    args.num_user = DATASET_CONFIG[args.dataset]["num_user"]
    args.num_item = DATASET_CONFIG[args.dataset]["num_item"]
    args.data_path = str(Path("datasets") / args.dataset) + os.sep
    args.device = torch.device(args.device)

    np.random.seed(2024)
    torch.manual_seed(2024)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(2024)

    record_path = Path("saved") / args.dataset / args.runid
    record_path.mkdir(parents=True, exist_ok=True)
    log = Logger(record_path)
    for name, value in vars(args).items():
        log.write(f"{name}={value}\n")

    dataset = Dataset(args)
    model = DCIL(args, dataset).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ndcg = -1.0
    best_state = None
    early_stop = 0
    for epoch in range(args.epochs):
        model.train()
        start = time()
        loss_sums = np.zeros(4, dtype=np.float64)
        batch_count = 0
        for batch_users, batch_items in tqdm(dataset.batch_sampling_softmax(), desc=f"epoch {epoch}"):
            users = torch.LongTensor(batch_users).to(args.device)
            items = torch.LongTensor(batch_items).to(args.device)
            optimizer.zero_grad()
            total_loss, rec_loss, rc_loss, pc_loss = model.get_loss(users, items)
            total_loss.backward()
            optimizer.step()
            loss_sums += np.array([total_loss.item(), rec_loss.item(), rc_loss.item(), pc_loss.item()])
            batch_count += 1
        loss_sums /= max(1, batch_count)
        log.write(
            f"Epoch:{epoch}, Loss:{loss_sums[0]:.4f}, Rec:{loss_sums[1]:.4f}, "
            f"RC:{loss_sums[2]:.4f}, PC:{loss_sums[3]:.4f}\n"
        )

        model.eval()
        user_emb, item_emb = model.get_embeddings()
        precision, recall, ndcg = ranking_metrics(
            dataset.valdata, dataset.traindata, [args.topk], user_emb, item_emb, dataset.valdata.keys()
        )
        log.write(
            f"Epoch:{epoch}, topk:{args.topk}, R@{args.topk}:{recall[args.topk]:.4f}, "
            f"P@{args.topk}:{precision[args.topk]:.4f}, N@{args.topk}:{ndcg[args.topk]:.4f}, "
            f"time:{time() - start:.2f}s\n\n"
        )
        if ndcg[args.topk] > best_ndcg:
            best_ndcg = ndcg[args.topk]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            early_stop = 0
            torch.save(best_state, record_path / "best_dcil.pt")
        else:
            early_stop += 1
        if epoch > 30 and early_stop > args.early_stops:
            log.write("early stop\n")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    user_emb, item_emb = model.get_embeddings()
    precision, recall, ndcg = ranking_metrics(
        dataset.testdata, dataset.traindata, [10, 20, 30, 40, 50], user_emb, item_emb, dataset.testdata.keys()
    )
    log.write("=================Evaluation results==================\n")
    for k in sorted(ndcg):
        log.write(f"Topk:{k:3d}, R@{k}:{recall[k]:.4f}, P@{k}:{precision[k]:.4f}, N@{k}:{ndcg[k]:.4f}\n")
    log.close()


if __name__ == "__main__":
    main()
