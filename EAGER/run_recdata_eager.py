#!/usr/bin/env python3
import argparse
import copy
import io
import os
import os.path as osp
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        rank[:, k - i - 1] = np.sum(rank[:, : k - i], axis=1)
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
        ground_truth, pred_topk = test[i], pred[i]
        hits = list(map(lambda x: x in ground_truth, pred_topk))
        hits = np.array(hits).astype("float")
        r.append(hits)
    return np.array(r).astype("float")


def load_recdata(data_root, dataset):
    train_file = osp.join(data_root, dataset, f"{dataset}_train.txt")
    test_file = osp.join(data_root, dataset, f"{dataset}_test.txt")
    if not osp.exists(train_file):
        raise FileNotFoundError(train_file)
    if not osp.exists(test_file):
        raise FileNotFoundError(test_file)

    train_samples = []
    val_data = {}
    test_data = {}
    neg_items = {}
    n_user = 0
    max_item = 0

    with open(train_file, "r") as f:
        for line in f:
            arr = line.strip().split(" ")
            if len(arr) < 2:
                continue
            user = int(arr[0]) - 1
            items = [int(x) - 1 for x in arr[1:]]
            n_user = max(n_user, user + 1)
            max_item = max(max_item, max(items) + 1)

            if len(items) >= 3:
                train_hist = items[:-2]
                for t in range(len(train_hist)):
                    train_samples.append((user, train_hist[:t], train_hist[t]))
                val_data[user] = [train_hist, items[-2]]
                test_data[user] = [train_hist, items[-1]]
            else:
                for t in range(len(items)):
                    train_samples.append((user, items[:t], items[t]))
                val_data[user] = []
                test_data[user] = []

    with open(test_file, "r") as f:
        for line in f:
            arr = line.strip().split(" ")
            if len(arr) < 2:
                continue
            user = int(arr[0]) - 1
            items = [int(x) - 1 for x in arr[1:]]
            if len(items) > 0:
                max_item = max(max_item, max(items) + 1)
            neg_items[user] = items

    return {
        "train_samples": train_samples,
        "val_data": val_data,
        "test_data": test_data,
        "neg_items": neg_items,
        "n_user": n_user,
        "item_num": max_item,
    }


def load_sasrec_embed(data_root, dataset, item_num):
    embed_file = osp.join(data_root, dataset, "SASRec_item_embed.pkl")
    if not osp.exists(embed_file):
        raise FileNotFoundError(embed_file)
    try:
        with open(embed_file, "rb") as f:
            emb = pickle.load(f)
    except RuntimeError as e:
        # Some files are pickled with CUDA storages (e.g., cuda:1).
        # Force both outer and nested tensor storages to CPU.
        if "Attempting to deserialize object on CUDA device" in str(e):
            original_loader = torch.storage._load_from_bytes
            try:
                torch.storage._load_from_bytes = (
                    lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
                )
                with open(embed_file, "rb") as f:
                    emb = pickle.load(f)
            finally:
                torch.storage._load_from_bytes = original_loader
        else:
            raise

    emb = torch.as_tensor(emb, dtype=torch.float32)
    if emb.shape[0] < item_num:
        raise ValueError(f"SASRec embedding rows {emb.shape[0]} < item_num {item_num}")
    if emb.shape[0] > item_num:
        emb = emb[:item_num]
    return emb


def pad_history(history, max_len, pad_id):
    h = history[-max_len:]
    if len(h) < max_len:
        h = [pad_id] * (max_len - len(h)) + h
    return h


def build_train_tensors(train_samples, seq_len, pad_id):
    x, y = [], []
    hist_len = seq_len - 1
    for _, hist, label in train_samples:
        x.append(pad_history(hist, hist_len, pad_id))
        y.append(label)
    return torch.LongTensor(x), torch.LongTensor(y)


def evaluate_split(model, data, users, seq_len, pad_id, topk, split_name):
    max_k = max(topk)
    results = {
        "Precision": np.zeros(len(topk)),
        "Recall": np.zeros(len(topk)),
        "MRR": np.zeros(len(topk)),
        "MAP": np.zeros(len(topk)),
        "NDCG": np.zeros(len(topk)),
    }

    valid_users = 0
    with torch.no_grad():
        for u in users:
            target_data = data["test_data"] if split_name == "test" else data["val_data"]
            if len(target_data.get(u, [])) == 0:
                continue
            if u not in data["neg_items"]:
                continue

            history, gt = target_data[u]
            cands = [gt] + data["neg_items"][u]
            if len(cands) < max_k:
                continue

            hx = pad_history(history, seq_len - 1, pad_id)
            input_x = torch.LongTensor([hx] * len(cands)).cuda()
            input_y = torch.LongTensor(cands).cuda()
            scores = model.compute_scores(input_x, input_y, type=0).sum(-1)
            pred_idx = torch.topk(scores, k=max_k).indices.detach().cpu().numpy().tolist()

            ground_truth = [0]
            r = getLabel([ground_truth], [pred_idx])
            for j, k in enumerate(topk):
                pre, rec = RecallPrecision_atK([ground_truth], r, k)
                mrr = MRR_atK([ground_truth], r, k)
                map_score = MAP_atK([ground_truth], r, k)
                ndcg = NDCG_atK([ground_truth], r, k)
                results["Precision"][j] += pre
                results["Recall"][j] += rec
                results["MRR"][j] += mrr
                results["MAP"][j] += map_score
                results["NDCG"][j] += ndcg
            valid_users += 1

    if valid_users == 0:
        raise RuntimeError(f"No valid users evaluated in split: {split_name}")
    for key in results:
        results[key] /= float(valid_users)
    return results, valid_users


def main():
    parser = argparse.ArgumentParser("EAGER + RecData pipeline")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--eager_root", type=str, default="/u/hchen42/EAGER/EAGER")
    parser.add_argument("--output_dir", type=str, default="/u/hchen42/EAGER/results")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_eval_users", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--enc_num_layers", type=int, default=1)
    parser.add_argument("--dec_num_layers", type=int, default=2)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--feature_ratio", type=float, default=1.0)
    parser.add_argument("--topk", type=str, default="1,5,10,20,100")
    parser.add_argument("--select_metric", type=str, default="NDCG")
    parser.add_argument("--select_k", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("EAGER implementation requires CUDA in this setup.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"
    set_seed(args.seed)

    sys.path.insert(0, args.eager_root)
    from lib import Trm4Rec  # noqa: E402

    data = load_recdata(args.data_path, args.dataset)
    item_num = data["item_num"]
    sasrec_emb = load_sasrec_embed(args.data_path, args.dataset, item_num)
    feat_dim = int(sasrec_emb.shape[1])

    train_x, train_y = build_train_tensors(data["train_samples"], args.seq_len, item_num)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    run_name = f"eager_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = osp.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    item_to_code_file = osp.join(out_dir, "item_to_code.npy")
    code_to_item_file = osp.join(out_dir, "code_to_item.npy")

    model = Trm4Rec(
        item_num=item_num,
        user_seq_len=args.seq_len - 1,
        d_model=args.d_model,
        d_model2=feat_dim,
        nhead=args.n_head,
        device=device,
        optimizer=lambda p: torch.optim.Adam(p, lr=args.lr, amsgrad=True),
        enc_num_layers=args.enc_num_layers,
        dec_num_layers=args.dec_num_layers,
        k=args.k,
        item_to_code_file=item_to_code_file,
        code_to_item_file=code_to_item_file,
        tree_has_generated=False,
        init_way="embkm",
        data=sasrec_emb,
        max_iters=args.max_iters,
        feature_ratio=args.feature_ratio,
        parall=4,
        type=0,
    )
    model.trm_model.train()

    topk = [int(k) for k in args.topk.split(",")]
    if args.select_metric not in {"Precision", "Recall", "MRR", "MAP", "NDCG"}:
        raise ValueError("select_metric must be one of Precision/Recall/MRR/MAP/NDCG")
    if args.select_k not in topk:
        raise ValueError("select_k must be included in topk")
    select_idx = topk.index(args.select_k)

    step = 0
    best_score = -1.0
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    users = sorted(list(data["test_data"].keys()))
    if args.max_eval_users > 0:
        users = users[: args.max_eval_users]

    for ep in range(1, args.epochs + 1):
        model.trm_model.train()
        running_loss = 0.0
        running_steps = 0
        for batch_x, batch_y in tqdm(train_loader, desc="Train"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss, _, _ = model.update_model(
                batch_x,
                batch_y,
                data_emb=sasrec_emb,
                type=0,
                use_con=False,
                use_guide=False,
                guide_feat=None,
            )
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            running_loss += float(loss.item())
            running_steps += 1
            step += 1
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break
        model.trm_model.eval()
        val_results, val_users = evaluate_split(
            model,
            data,
            users,
            args.seq_len,
            item_num,
            topk,
            split_name="val",
        )
        score = float(val_results[args.select_metric][select_idx])
        avg_loss = running_loss / max(running_steps, 1)
        print(
            f"Epoch {ep}/{args.epochs} - train_loss={avg_loss:.6f}, "
            f"val_{args.select_metric}@{args.select_k}={score:.6f}, val_users={val_users}"
        )

        if score > best_score:
            best_score = score
            best_epoch = ep
            best_state = copy.deepcopy(model.trm_model.state_dict())
            bad_epochs = 0
            print(f"New best checkpoint at epoch {ep} ({args.select_metric}@{args.select_k}={score:.6f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {ep}, no improvement for {args.patience} epochs.")
                break

        if args.max_train_steps > 0 and step >= args.max_train_steps:
            print("Reached max_train_steps, stopping training.")
            break

    if best_state is not None:
        model.trm_model.load_state_dict(best_state, strict=True)
        print(
            f"Loaded best checkpoint from epoch {best_epoch} "
            f"({args.select_metric}@{args.select_k}={best_score:.6f})"
        )

    model.trm_model.eval()
    results, valid_users = evaluate_split(
        model,
        data,
        users,
        args.seq_len,
        item_num,
        topk,
        split_name="test",
    )

    print(f"Dataset: {args.dataset}")
    print(f"Evaluated users: {valid_users}")
    for j, k in enumerate(topk):
        print(
            f"@{k}: Precision={results['Precision'][j]:.6f}, "
            f"Recall={results['Recall'][j]:.6f}, "
            f"MRR={results['MRR'][j]:.6f}, "
            f"MAP={results['MAP'][j]:.6f}, "
            f"NDCG={results['NDCG'][j]:.6f}"
        )

    np.savez(osp.join(out_dir, "results.npz"), topk=np.array(topk), **results)
    with open(osp.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"dataset={args.dataset}\n")
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_{args.select_metric}@{args.select_k}={best_score:.6f}\n")
        f.write(f"valid_users={valid_users}\n")
        for j, k in enumerate(topk):
            f.write(
                f"@{k}: Precision={results['Precision'][j]:.6f}, "
                f"Recall={results['Recall'][j]:.6f}, "
                f"MRR={results['MRR'][j]:.6f}, "
                f"MAP={results['MAP'][j]:.6f}, "
                f"NDCG={results['NDCG'][j]:.6f}\n"
            )
    torch.save(model.trm_model.state_dict(), osp.join(out_dir, "eager_model.pt"))
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
