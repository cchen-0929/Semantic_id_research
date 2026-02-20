import io
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Tuple

import torch

from data.schemas import SeqBatch


def _safe_load_pickle_tensor(path: str) -> torch.Tensor:
    """
    Load tensor serialized in pickle/torch format, forcing CPU map location.
    """
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        with open(path, "rb") as f:
            # Handle nested torch storages saved with CUDA device.
            torch.storage._load_from_bytes = lambda b: torch.load(
                io.BytesIO(b), map_location="cpu", weights_only=False
            )
            obj = pickle.load(f)
    if isinstance(obj, torch.Tensor):
        return obj
    return torch.tensor(obj)


@dataclass
class RecDataRecord:
    user_id: int
    train_history: List[int]
    val_gt: int
    test_gt: int
    test_negatives: List[int]


def _load_recdata_records(data_root: str, split: str) -> Tuple[List[RecDataRecord], int, int]:
    dataset_dir = os.path.join(data_root, split)
    train_file = os.path.join(dataset_dir, f"{split}_train.txt")
    test_file = os.path.join(dataset_dir, f"{split}_test.txt")

    test_negs: Dict[int, List[int]] = {}
    max_item = 0
    max_user = 0
    with open(test_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                continue
            user = int(parts[0]) - 1
            negs = [int(x) for x in parts[1:]]
            test_negs[user] = negs
            if negs:
                max_item = max(max_item, max(negs))
            max_user = max(max_user, user)

    out = []
    with open(train_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 4:
                continue
            user = int(parts[0]) - 1
            items = [int(x) for x in parts[1:]]
            train_history = items[:-2]
            val_gt = items[-2]
            test_gt = items[-1]
            negatives = test_negs.get(user, [])
            out.append(
                RecDataRecord(
                    user_id=user,
                    train_history=train_history,
                    val_gt=val_gt,
                    test_gt=test_gt,
                    test_negatives=negatives,
                )
            )
            max_item = max(max_item, max(items))
            max_user = max(max_user, user)
    return out, max_user + 1, max_item + 1


class RecDataItemData(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str, train_test_split: str = "all", train_eval_ratio: float = 0.95):
        self.records, self.n_users, self.n_items = _load_recdata_records(root, split)
        feat_path = os.path.join(root, split, "SASRec_item_embed.pkl")
        self.item_data = _safe_load_pickle_tensor(feat_path).to(torch.float32)
        # Keep one text field for compatibility.
        self.item_text = [f"item_{i}" for i in range(self.item_data.shape[0])]

        all_ids = torch.arange(self.item_data.shape[0], dtype=torch.long)
        if train_test_split == "all":
            self.item_ids = all_ids
        else:
            gen = torch.Generator()
            gen.manual_seed(42)
            perm = all_ids[1:][torch.randperm(len(all_ids) - 1, generator=gen)]
            split_idx = int(len(perm) * train_eval_ratio)
            if train_test_split == "train":
                self.item_ids = torch.cat([torch.tensor([0]), perm[:split_idx]])
            elif train_test_split == "eval":
                self.item_ids = perm[split_idx:]
            else:
                raise ValueError(f"Unsupported train_test_split: {train_test_split}")

    def __len__(self):
        return self.item_ids.shape[0]

    def __getitem__(self, idx):
        item_ids = self.item_ids[idx] if isinstance(idx, torch.Tensor) else self.item_ids[torch.tensor(idx)]
        if item_ids.ndim == 0:
            item_ids = item_ids.unsqueeze(0)
        x = self.item_data[item_ids]
        return SeqBatch(
            user_ids=-1 * torch.ones_like(item_ids),
            ids=item_ids,
            ids_fut=-1 * torch.ones_like(item_ids),
            x=x,
            x_fut=-1 * torch.ones_like(x),
            seq_mask=torch.ones_like(item_ids, dtype=torch.bool),
        )


class RecDataSeqData(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str, is_train: bool = True, subsample: bool = False, max_seq_len: int = 50):
        self.records, self.n_users, self.n_items = _load_recdata_records(root, split)
        feat_path = os.path.join(root, split, "SASRec_item_embed.pkl")
        self.item_data = _safe_load_pickle_tensor(feat_path).to(torch.float32)
        self._max_seq_len = max_seq_len
        self.is_train = is_train
        self.subsample = subsample
        self._eval_records = [r for r in self.records if len(r.test_negatives) > 0]

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @property
    def eval_records(self):
        return self._eval_records

    def __len__(self):
        return len(self.records)

    def _pad_history(self, history: List[int]) -> torch.Tensor:
        history = history[-self.max_seq_len :]
        out = torch.tensor(history, dtype=torch.long)
        if out.shape[0] < self.max_seq_len:
            out = torch.cat([out, -1 * torch.ones(self.max_seq_len - out.shape[0], dtype=torch.long)], dim=0)
        return out

    def __getitem__(self, idx):
        rec = self.records[idx]
        user_id = torch.tensor(rec.user_id, dtype=torch.long)

        if self.is_train:
            # Train on validation target to avoid leaking test gt into training.
            if self.subsample:
                seq = rec.train_history + [rec.val_gt]
                end = random.randint(2, len(seq))
                history = seq[max(0, end - 1 - self.max_seq_len) : end - 1]
                target = seq[end - 1]
            else:
                history = rec.train_history
                target = rec.val_gt
        else:
            history = rec.train_history
            target = rec.test_gt

        item_ids = self._pad_history(history)
        item_ids_fut = torch.tensor([target], dtype=torch.long)
        x = self.item_data[item_ids]
        x[item_ids == -1] = -1
        x_fut = self.item_data[item_ids_fut]
        x_fut[item_ids_fut == -1] = -1
        return SeqBatch(
            user_ids=user_id,
            ids=item_ids,
            ids_fut=item_ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=(item_ids >= 0),
        )

    def build_candidate_batch(self, user_id: int, history: List[int], candidates: List[int]) -> SeqBatch:
        hist = self._pad_history(history)
        ids = hist.unsqueeze(0).repeat(len(candidates), 1)
        ids_fut = torch.tensor(candidates, dtype=torch.long).unsqueeze(1)
        users = torch.full((len(candidates),), fill_value=user_id, dtype=torch.long)
        x = self.item_data[ids]
        x[ids == -1] = -1
        x_fut = self.item_data[ids_fut]
        return SeqBatch(
            user_ids=users,
            ids=ids,
            ids_fut=ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=(ids >= 0),
        )
