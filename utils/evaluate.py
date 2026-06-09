import numpy as np


def _dcg(items):
    return sum(1.0 / np.log2(rank + 2) for rank, hit in enumerate(items) if hit)


def ranking_metrics(testdata, traindata, topk_list, user_emb, item_emb, test_users=None):
    """Full-ranking Recall, Precision, and NDCG evaluation."""
    if test_users is None:
        test_users = list(testdata.keys())
    max_k = max(topk_list)
    precision = {k: 0.0 for k in topk_list}
    recall = {k: 0.0 for k in topk_list}
    ndcg = {k: 0.0 for k in topk_list}
    evaluated = 0

    for user in test_users:
        positives = set(testdata.get(user, []))
        if not positives:
            continue
        scores = np.matmul(item_emb, user_emb[user])
        for item in traindata.get(user, []):
            scores[item] = -np.inf
        candidates = np.argpartition(-scores, max_k - 1)[:max_k]
        candidates = candidates[np.argsort(-scores[candidates])]
        hits = [item in positives for item in candidates]
        evaluated += 1
        for k in topk_list:
            top_hits = hits[:k]
            hit_count = sum(top_hits)
            precision[k] += hit_count / k
            recall[k] += hit_count / len(positives)
            ideal = _dcg([True] * min(k, len(positives)))
            ndcg[k] += _dcg(top_hits) / ideal if ideal > 0 else 0.0

    if evaluated == 0:
        return precision, recall, ndcg
    for metric in (precision, recall, ndcg):
        for k in topk_list:
            metric[k] /= evaluated
    return precision, recall, ndcg
