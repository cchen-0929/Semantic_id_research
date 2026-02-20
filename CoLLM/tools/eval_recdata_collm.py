import argparse
import json
import os

import numpy as np
import torch

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.datasets.datasets.rec_gnndataset import RecDataGnnDataset
from minigpt4.datasets.datasets.recdata_adapter import load_recdata

# registration side effects
from minigpt4.datasets.builders import *  # noqa: F401,F403
from minigpt4.models import *  # noqa: F401,F403
from minigpt4.processors import *  # noqa: F401,F403
from minigpt4.runners import *  # noqa: F401,F403
from minigpt4.tasks import *  # noqa: F401,F403


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CoLLM on RecData Top-K metrics")
    parser.add_argument("--cfg-path", required=True, help="training/eval config yaml")
    parser.add_argument("--data-path", required=True, help="RecData dataset dir, e.g. .../RecData/Beauty")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", default="1,5,10,20,100")
    parser.add_argument("--batch-candidates", type=int, default=128)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--output", default="")
    parser.add_argument("--options", nargs="+", default=None)
    return parser.parse_args()


def RecallPrecision_atK(test, r, k):
    tp = r[:, :k].sum(1)
    precision = np.sum(tp) / (k * len(test))
    recall_n = np.array([len(test[i]) for i in range(len(test))])
    recall = np.sum(tp / recall_n) / len(test)
    return precision, recall


def MRR_atK(test, r, k):
    pred = r[:, :k]
    weight = np.arange(1, k + 1)
    mrr = np.sum(pred / weight, axis=1) / np.array([
        len(test[i]) if len(test[i]) <= k else k for i in range(len(test))
    ])
    return np.sum(mrr) / len(test)


def MAP_atK(test, r, k):
    pred = r[:, :k]
    rank = pred.copy()
    for i in range(k):
        rank[:, k - i - 1] = np.sum(rank[:, : k - i], axis=1)
    weight = np.arange(1, k + 1)
    ap = np.sum(pred * rank / weight, axis=1)
    ap = ap / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    return np.sum(ap) / len(test)


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
    return np.sum(ndcg) / len(test)


def getLabel(test, pred):
    r = []
    for i in range(len(test)):
        gt, pred_topk = test[i], pred[i]
        hits = list(map(lambda x: x in gt, pred_topk))
        r.append(np.array(hits).astype("float"))
    return np.array(r).astype("float")


def _build_candidate_batch(model, device, user, history, candidates, max_history, sas_seq_len):
    hist = history[-max_history:]
    padded_hist = [0] * (max_history - len(hist)) + hist
    hist_titles = [f'"item_{x}"' for x in hist if x > 0]
    hist_titles = ", ".join(hist_titles) if len(hist_titles) > 0 else "unkow"

    user_ids = torch.full((len(candidates),), fill_value=user, dtype=torch.long, device=device)
    target_ids = torch.tensor(candidates, dtype=torch.long, device=device)
    labels = torch.ones((len(candidates),), dtype=torch.long, device=device)
    his_pad = torch.tensor(np.repeat(np.array([padded_hist]), len(candidates), axis=0), dtype=torch.long, device=device)

    sample = {
        "UserID": user_ids,
        "TargetItemID": target_ids,
        "label": labels,
        "InteractedItemIDs_pad": his_pad,
        "InteractedNum": [len(hist)] * len(candidates),
        "InteractedItemTitles": [hist_titles] * len(candidates),
        "TargetItemTitle": [f'"item_{x}"' for x in candidates],
    }

    if model.rec_model_type in ["sasrec", "DIN"]:
        sas_hist = history[-sas_seq_len:]
        sas_padded = [0] * (sas_seq_len - len(sas_hist)) + sas_hist
        sas_seq = torch.tensor(np.repeat(np.array([sas_padded]), len(candidates), axis=0), dtype=torch.long, device=device)
        sample["sas_seq"] = sas_seq

    return sample


def _infer_user_item_num(cfg, recdata):
    cfg.model_cfg.rec_config.user_num = int(recdata.n_user)
    cfg.model_cfg.rec_config.item_num = int(recdata.m_item)


def main():
    args = parse_args()
    cfg = Config(args)

    recdata = load_recdata(args.data_path)
    _infer_user_item_num(cfg, recdata)

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)

    device = torch.device(args.device)
    model = model.to(device)
    model.set_mode(cfg.run_cfg.get("mode", "v2"))
    model.eval()

    if cfg.model_cfg.rec_model == "lightgcn":
        gnn_data = RecDataGnnDataset(cfg.model_cfg.rec_config, args.data_path)
        model.rec_encoder._set_graph(gnn_data.Graph)

    topk = [int(x) for x in args.topk.split(",")]
    max_k = max(topk)

    metrics = {
        "Precision": np.zeros(len(topk), dtype=np.float64),
        "Recall": np.zeros(len(topk), dtype=np.float64),
        "MRR": np.zeros(len(topk), dtype=np.float64),
        "MAP": np.zeros(len(topk), dtype=np.float64),
        "NDCG": np.zeros(len(topk), dtype=np.float64),
    }

    test_users = []
    for user, rec in recdata.records.items():
        if rec.test_gt is None or len(rec.test_negatives) == 0:
            continue
        test_users.append((user, rec.train_history, rec.test_gt, rec.test_negatives))

    if len(test_users) == 0:
        raise RuntimeError(f"No valid test users found in {args.data_path}")

    with torch.no_grad():
        valid_users = 0
        for user, history, gt, negatives in test_users:
            candidates = [gt] + negatives
            if len(candidates) < max_k:
                continue
            scores = []

            for st in range(0, len(candidates), args.batch_candidates):
                chunk = candidates[st : st + args.batch_candidates]
                sample = _build_candidate_batch(
                    model=model,
                    device=device,
                    user=user,
                    history=history,
                    candidates=chunk,
                    max_history=args.max_history,
                    sas_seq_len=int(cfg.model_cfg.rec_config.get("maxlen", 20)),
                )
                out = model.generate_for_samples(sample)
                logits = out["logits"].detach().float().cpu().numpy()
                scores.extend(logits.tolist())

            scores = np.array(scores)
            top_idx = np.argsort(-scores)[:max_k]
            pred = [top_idx]
            gt_idx = [[0]]
            r = getLabel(gt_idx, pred)

            for j, k in enumerate(topk):
                pre, rec = RecallPrecision_atK(gt_idx, r, k)
                mrr = MRR_atK(gt_idx, r, k)
                map_score = MAP_atK(gt_idx, r, k)
                ndcg = NDCG_atK(gt_idx, r, k)
                metrics["Precision"][j] += pre
                metrics["Recall"][j] += rec
                metrics["MRR"][j] += mrr
                metrics["MAP"][j] += map_score
                metrics["NDCG"][j] += ndcg
            valid_users += 1

    if valid_users == 0:
        raise RuntimeError("No users have enough candidates for requested top-k.")

    denom = float(valid_users)
    for key in metrics:
        metrics[key] /= denom

    result = {
        "dataset": os.path.basename(args.data_path.rstrip("/")),
        "valid_users": valid_users,
        "topk": topk,
        "metrics": {k: metrics[k].tolist() for k in metrics},
    }

    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
