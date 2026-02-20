import math


def _ndcg(hit_rank):
    if hit_rank < 0:
        return 0.0
    return 1.0 / math.log2(hit_rank + 2.0)


def evaluate_one_ranklist(ranked_items, gt, topk):
    out = {}
    hit_rank = -1
    for i, x in enumerate(ranked_items):
        if x == gt:
            hit_rank = i
            break

    for k in topk:
        if hit_rank >= 0 and hit_rank < k:
            p = 1.0 / float(k)
            r = 1.0
            mrr = 1.0 / float(hit_rank + 1)
            ap = mrr
            ndcg = _ndcg(hit_rank)
        else:
            p = 0.0
            r = 0.0
            mrr = 0.0
            ap = 0.0
            ndcg = 0.0
        out[k] = {
            "Precision": p,
            "Recall": r,
            "MRR": mrr,
            "MAP": ap,
            "NDCG": ndcg,
        }
    return out

