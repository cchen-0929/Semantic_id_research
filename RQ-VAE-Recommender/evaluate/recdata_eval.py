import numpy as np
import torch
import torch.nn.functional as F

from data.utils import batch_to


def RecallPrecision_atK(test, r, k):
    tp = r[:, :k].sum(1)
    precision = np.sum(tp) / (k * len(test))
    recall_n = np.array([len(test[i]) for i in range(len(test))])
    recall = np.sum(tp / recall_n) / len(test)
    return precision, recall


def MRR_atK(test, r, k):
    pred = r[:, :k]
    weight = np.arange(1, k + 1)
    mrr = np.sum(pred / weight, axis=1) / np.array(
        [len(test[i]) if len(test[i]) <= k else k for i in range(len(test))]
    )
    mrr = np.sum(mrr) / len(test)
    return mrr


def MAP_atK(test, r, k):
    pred = r[:, :k]
    rank = pred.copy()
    for i in range(k):
        rank[:, k - i - 1] = np.sum(rank[:, :k - i], axis=1)
    weight = np.arange(1, k + 1)
    ap = np.sum(pred * rank / weight, axis=1)
    ap = ap / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    map_score = np.sum(ap) / len(test)
    return map_score


def NDCG_atK(test, r, k):
    pred = r[:, :k]
    test_mat = np.zeros((len(pred), k))
    for i, items in enumerate(test):
        length = k if k <= len(items) else len(items)
        test_mat[i, :length] = 1

    idcg = np.sum(test_mat * (1.0 / np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.0] = 1.0
    dcg = pred * (1.0 / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.0
    ndcg = np.sum(ndcg) / len(test)
    return ndcg


def getLabel(test, pred):
    r = []
    for i in range(len(test)):
        groundTruth, predTopK = test[i], pred[i]
        hits = list(map(lambda x: x in groundTruth, predTopK))
        hits = np.array(hits).astype("float")
        r.append(hits)
    return np.array(r).astype("float")


@torch.no_grad()
def evaluate_recdata(model, tokenizer, eval_dataset, device, ks=(1, 5, 10, 20, 100)):
    model_was_training = model.training
    model.eval()
    model.enable_generation = False

    if not hasattr(eval_dataset, "_recdata"):
        raise ValueError("RecData evaluation requires eval_dataset to be backed by RecDataSeqData.")
    recdata = eval_dataset._recdata
    records = recdata.eval_records

    results = {
        "Precision": np.zeros(len(ks)),
        "Recall": np.zeros(len(ks)),
        "MRR": np.zeros(len(ks)),
        "MAP": np.zeros(len(ks)),
        "NDCG": np.zeros(len(ks)),
    }

    valid_users = 0
    max_k = max(ks)
    for rec in records:
        candidates = [rec.test_gt] + rec.test_negatives
        if len(candidates) < max_k:
            continue

        seq_batch = recdata.build_candidate_batch(
            user_id=rec.user_id,
            history=rec.train_history,
            candidates=candidates,
        )
        seq_batch = batch_to(seq_batch, device)
        tokenized = tokenizer(seq_batch)
        model_output = model(tokenized)
        logits = model_output.logits
        targets = tokenized.sem_ids_fut.flatten()
        # Candidate-level NLL score (higher is better after negation).
        loss = F.cross_entropy(logits, targets, reduction="none", ignore_index=-1)
        loss = loss.view(len(candidates), -1).sum(dim=1)
        scores = -loss

        top_idx = torch.topk(scores, k=max_k).indices.cpu().numpy()
        ground_truth = [[0]]
        r = getLabel(ground_truth, [top_idx])

        for j, k in enumerate(ks):
            pre, rec_score = RecallPrecision_atK(ground_truth, r, k)
            mrr = MRR_atK(ground_truth, r, k)
            map_score = MAP_atK(ground_truth, r, k)
            ndcg = NDCG_atK(ground_truth, r, k)
            results["Precision"][j] += pre
            results["Recall"][j] += rec_score
            results["MRR"][j] += mrr
            results["MAP"][j] += map_score
            results["NDCG"][j] += ndcg
        valid_users += 1

    if valid_users > 0:
        for key in results.keys():
            results[key] /= float(valid_users)

    model.train(model_was_training)
    return results
