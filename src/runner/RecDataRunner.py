import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from data.recdata_adapter import load_recdata
from utils.prompt import load_prompt_template
from utils.recdata_metrics import evaluate_one_ranklist


class RecDataRunner:
    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.rank = getattr(args, "rank", 0)
        self.world_size = getattr(args, "world_size", 1)
        self.is_distributed = self.world_size > 1 and dist.is_initialized()

        self.topk = [int(x) for x in str(args.topk).split(",")]
        self.max_k = max(self.topk)

    def _item_token(self, item_id):
        if self.args.his_prefix > 0:
            return f"item_{item_id}"
        return str(item_id)

    def _format_history(self, history):
        if self.args.his_prefix > 0:
            seq = [f"item_{x}" for x in history]
        else:
            seq = [str(x) for x in history]
        if self.args.max_his > 0:
            seq = seq[-self.args.max_his :]
        return self.args.his_sep.join(seq)

    def _build_texts(self, dataset, user, history, target_token):
        tasks = self.args.tasks.split(",")
        if "sequential" not in tasks:
            raise ValueError("RecData mode expects task 'sequential'.")

        prompt_templates = load_prompt_template(self.args.prompt_file, ["sequential"])
        mode, pid = self.args.test_prompt.split(":")
        prompt = prompt_templates["sequential"][mode][pid]

        data = {
            "dataset": dataset,
            "user_id": str(user),
            "history": self._format_history(history),
            "target": target_token,
        }
        inp = prompt["Input"].format(**data)
        out = prompt["Output"].format(**data)
        return inp, out

    def _score_candidates(self, input_text, candidate_texts, chunk_size=64):
        enc = self.tokenizer(
            [input_text],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        scores = []
        for s in range(0, len(candidate_texts), chunk_size):
            part = candidate_texts[s : s + chunk_size]
            dec = self.tokenizer(
                part,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            labels = dec["input_ids"].to(self.device)
            labels_mask = (labels != self.tokenizer.pad_token_id).long()
            labels[labels == self.tokenizer.pad_token_id] = -100

            bsz = labels.size(0)
            out = self.model(
                input_ids=input_ids.repeat(bsz, 1),
                attention_mask=attention_mask.repeat(bsz, 1),
                labels=labels,
                return_dict=True,
            )
            logits = out.logits
            logp = F.log_softmax(logits, dim=-1)

            safe_labels = labels.clone()
            safe_labels[safe_labels < 0] = 0
            tok_logp = logp.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
            tok_logp = tok_logp * labels_mask
            denom = labels_mask.sum(dim=1).clamp(min=1)
            seq_score = tok_logp.sum(dim=1) / denom
            scores.extend(seq_score.detach().cpu().tolist())
        return scores

    def _reduce_metrics(self, sums, cnt):
        if not self.is_distributed:
            return sums, cnt
        t = torch.tensor(sums + [float(cnt)], device=self.device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out_cnt = int(t[-1].item())
        out_sums = [x for x in t[:-1].tolist()]
        return out_sums, out_cnt

    def _eval_split(self, dataset, split_data, all_items, neg_items):
        metric_names = ["Precision", "Recall", "MRR", "MAP", "NDCG"]
        sums = [0.0] * (len(self.topk) * len(metric_names))
        count = 0

        if self.rank == 0:
            bar = tqdm(total=len(split_data), desc=f"Eval {dataset}", disable=False)
        else:
            bar = None

        for idx, (user, history, gt) in enumerate(split_data):
            if idx % self.world_size != self.rank:
                continue

            neg = neg_items.get(user, [])
            if len(neg) == 0:
                seen = set(history)
                neg = [x for x in all_items if x != gt and x not in seen]

            candidates = [gt] + neg
            if len(candidates) < self.max_k:
                continue

            cand_tokens = [self._item_token(x) for x in candidates]
            input_text, _ = self._build_texts(dataset, user, history, cand_tokens[0])

            cand_outputs = []
            for c in cand_tokens:
                _, out_text = self._build_texts(dataset, user, history, c)
                cand_outputs.append(out_text)

            scores = self._score_candidates(input_text, cand_outputs)
            ranked_idx = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            ranked_items = [candidates[i] for i in ranked_idx[: self.max_k]]
            one = evaluate_one_ranklist(ranked_items, gt, self.topk)

            for k_i, k in enumerate(self.topk):
                for m_i, m in enumerate(metric_names):
                    sums[k_i * len(metric_names) + m_i] += one[k][m]
            count += 1

            if bar is not None:
                bar.update(1)

        if bar is not None:
            bar.close()

        sums, count = self._reduce_metrics(sums, count)
        if count == 0:
            raise RuntimeError(f"No valid users in split for {dataset}")

        results = {}
        for k_i, k in enumerate(self.topk):
            results[k] = {}
            for m_i, m in enumerate(metric_names):
                results[k][m] = sums[k_i * len(metric_names) + m_i] / float(count)
        return results, count

    def test(self):
        self.model.eval()
        datasets = self.args.datasets.split(",")
        os.makedirs(self.args.recdata_output_dir, exist_ok=True)
        for ds in datasets:
            data = load_recdata(self.args.data_path, ds)
            test_results, valid_users = self._eval_split(
                ds,
                data["test_samples"],
                data["all_items"],
                data["neg_items"],
            )

            if self.rank == 0:
                logging.info(f"Testing RecData {ds} on task sequential")
                for k in self.topk:
                    r = test_results[k]
                    logging.info(
                        f"@{k}: Precision={r['Precision']:.6f}, Recall={r['Recall']:.6f}, "
                        f"MRR={r['MRR']:.6f}, MAP={r['MAP']:.6f}, NDCG={r['NDCG']:.6f}"
                    )

                out_file = os.path.join(self.args.recdata_output_dir, f"{ds}_metrics.txt")
                with open(out_file, "w") as f:
                    f.write(f"dataset={ds}\n")
                    f.write(f"valid_users={valid_users}\n")
                    for k in self.topk:
                        r = test_results[k]
                        f.write(
                            f"@{k}: Precision={r['Precision']:.6f}, Recall={r['Recall']:.6f}, "
                            f"MRR={r['MRR']:.6f}, MAP={r['MAP']:.6f}, NDCG={r['NDCG']:.6f}\n"
                        )

