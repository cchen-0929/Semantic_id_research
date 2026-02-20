import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RecDataUserRecord:
    user_id: int
    full_items: List[int]
    train_history: List[int]
    val_gt: Optional[int]
    test_gt: Optional[int]
    test_negatives: List[int]


@dataclass
class RecDataParsed:
    records: Dict[int, RecDataUserRecord]
    n_user: int
    m_item: int
    all_items: List[int]


def _parse_user_items(line: str):
    arr = line.strip().split()
    if len(arr) < 2:
        return None, []
    user = int(arr[0]) - 1
    items = [int(x) for x in arr[1:]]
    return user, items


def load_recdata(dataset_dir: str) -> RecDataParsed:
    dataset_dir = dataset_dir.rstrip("/")
    dataset_name = os.path.basename(dataset_dir)
    train_file = os.path.join(dataset_dir, f"{dataset_name}_train.txt")
    test_file = os.path.join(dataset_dir, f"{dataset_name}_test.txt")

    if not os.path.exists(train_file):
        raise FileNotFoundError(train_file)
    if not os.path.exists(test_file):
        raise FileNotFoundError(test_file)

    records: Dict[int, RecDataUserRecord] = {}
    max_user = -1
    max_item = 0

    with open(train_file, "r") as f:
        for line in f:
            user, items = _parse_user_items(line)
            if user is None or len(items) == 0:
                continue

            max_user = max(max_user, user)
            max_item = max(max_item, max(items))

            if len(items) >= 3:
                train_history = items[:-2]
                val_gt = items[-2]
                test_gt = items[-1]
            elif len(items) == 2:
                train_history = items[:-1]
                val_gt = items[-1]
                test_gt = items[-1]
            else:
                train_history = []
                val_gt = items[0]
                test_gt = items[0]

            records[user] = RecDataUserRecord(
                user_id=user,
                full_items=items,
                train_history=train_history,
                val_gt=val_gt,
                test_gt=test_gt,
                test_negatives=[],
            )

    with open(test_file, "r") as f:
        for line in f:
            user, negatives = _parse_user_items(line)
            if user is None:
                continue

            max_user = max(max_user, user)
            if len(negatives) > 0:
                max_item = max(max_item, max(negatives))

            if user not in records:
                records[user] = RecDataUserRecord(
                    user_id=user,
                    full_items=[],
                    train_history=[],
                    val_gt=None,
                    test_gt=None,
                    test_negatives=negatives,
                )
            else:
                records[user].test_negatives = negatives

    all_items = list(range(1, max_item + 1))
    return RecDataParsed(
        records=records,
        n_user=max_user + 1,
        m_item=max_item + 1,
        all_items=all_items,
    )
