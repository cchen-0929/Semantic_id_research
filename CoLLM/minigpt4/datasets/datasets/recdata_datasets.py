import random
from typing import List

import numpy as np

from minigpt4.datasets.datasets.rec_base_dataset import RecBaseDataset
from minigpt4.datasets.datasets.recdata_adapter import load_recdata


def _format_titles(ids: List[int]) -> str:
    titles = [f'"item_{x}"' for x in ids if x > 0]
    if len(titles) == 0:
        return "unkow"
    return ", ".join(titles)


class _RecDataBase(RecBaseDataset):
    def __init__(
        self,
        text_processor=None,
        ann_paths=None,
        max_history=10,
        sas_seq_len=20,
        with_sas_seq=False,
        train_neg_per_pos=1,
    ):
        super().__init__()
        if ann_paths is None or len(ann_paths) == 0:
            raise ValueError("ann_paths must be provided")

        spec = ann_paths[0]
        parts = spec.split("=")
        dataset_dir = parts[0]
        split = parts[1] if len(parts) > 1 else "train"

        parsed = load_recdata(dataset_dir)

        self.text_processor = text_processor
        self.with_sas_seq = with_sas_seq
        self.max_history = max_history
        self.sas_seq_len = sas_seq_len
        self.train_neg_per_pos = train_neg_per_pos

        self.user_num = parsed.n_user
        self.item_num = parsed.m_item
        self.all_items = parsed.all_items
        self.records = parsed.records

        self.annotation = self._build_samples(split)

        print(
            "RecData path:",
            dataset_dir,
            "split:",
            split,
            "samples:",
            len(self.annotation),
            "users:",
            self.user_num,
            "items:",
            self.item_num,
        )

    def _sample_negative(self, positive_items_set):
        if len(self.all_items) == 0:
            return 1
        for _ in range(50):
            item = random.choice(self.all_items)
            if item not in positive_items_set:
                return item
        for item in self.all_items:
            if item not in positive_items_set:
                return item
        return self.all_items[-1]

    def _add_eval_pair(self, samples, user_id, history, gt, negatives):
        if gt is None:
            return
        samples.append(
            {
                "UserID": user_id,
                "history": history,
                "TargetItemID": gt,
                "label": 1,
            }
        )
        neg = negatives[0] if len(negatives) > 0 else self._sample_negative(set(history + [gt]))
        samples.append(
            {
                "UserID": user_id,
                "history": history,
                "TargetItemID": neg,
                "label": 0,
            }
        )

    def _build_samples(self, split):
        samples = []
        for rec in self.records.values():
            hist = list(rec.train_history)
            if split == "train":
                if len(hist) == 0:
                    continue
                positives_set = set(rec.full_items)
                for t in range(1, len(hist) + 1):
                    prefix = hist[: t - 1]
                    pos_target = hist[t - 1]
                    samples.append(
                        {
                            "UserID": rec.user_id,
                            "history": prefix,
                            "TargetItemID": pos_target,
                            "label": 1,
                        }
                    )
                    for _ in range(self.train_neg_per_pos):
                        neg_target = self._sample_negative(positives_set)
                        samples.append(
                            {
                                "UserID": rec.user_id,
                                "history": prefix,
                                "TargetItemID": neg_target,
                                "label": 0,
                            }
                        )
            elif split == "valid":
                self._add_eval_pair(samples, rec.user_id, hist, rec.val_gt, rec.test_negatives)
            elif split == "test":
                self._add_eval_pair(samples, rec.user_id, hist, rec.test_gt, rec.test_negatives)
            else:
                raise ValueError(f"Unsupported split: {split}")
        return samples

    def __getitem__(self, index):
        ann = self.annotation[index]
        history = ann["history"]

        interacted_num = len(history)
        if len(history) < self.max_history:
            interacted_pad = [0] * (self.max_history - len(history)) + history
        else:
            interacted_pad = history[-self.max_history :]
            interacted_num = self.max_history

        sample = {
            "UserID": ann["UserID"],
            "InteractedItemIDs_pad": np.array(interacted_pad),
            "InteractedItemTitles": _format_titles(history[-interacted_num:]),
            "TargetItemID": ann["TargetItemID"],
            "TargetItemTitle": f'"item_{ann["TargetItemID"]}"',
            "InteractedNum": interacted_num,
            "label": ann["label"],
        }

        if self.with_sas_seq:
            if len(history) < self.sas_seq_len:
                sas_seq = [0] * (self.sas_seq_len - len(history)) + history
            else:
                sas_seq = history[-self.sas_seq_len :]
            sample["sas_seq"] = np.array(sas_seq)

        return sample


class RecDataOOData(_RecDataBase):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__(
            text_processor=text_processor,
            ann_paths=ann_paths,
            max_history=10,
            sas_seq_len=20,
            with_sas_seq=False,
            train_neg_per_pos=1,
        )


class RecDataOOData_sasrec(_RecDataBase):
    def __init__(self, text_processor=None, ann_paths=None, sas_seq_len=20):
        super().__init__(
            text_processor=text_processor,
            ann_paths=ann_paths,
            max_history=10,
            sas_seq_len=sas_seq_len,
            with_sas_seq=True,
            train_neg_per_pos=1,
        )
