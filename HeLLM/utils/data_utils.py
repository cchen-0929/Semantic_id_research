import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
import os

class SequentialDataset(Dataset):
    def __init__(self, dataset, maxlen, data_root="../Semantic_ID/RecData"):
        super(SequentialDataset, self).__init__()
        self.dataset_path = os.path.join(data_root, dataset)
        self.maxlen = maxlen

        self.trainData, self.valData, self.testData = [], {}, {}
        self.allPos = {}
        self.n_user, self.m_item = 0, 0

        train_file = os.path.join(self.dataset_path, f"{dataset}_train.txt")
        test_file = os.path.join(self.dataset_path, f"{dataset}_test.txt")

        with open(train_file, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                # RecData user ids are 1-based; convert to 0-based for internal indexing.
                user, items = int(line[0]) - 1, [int(item) for item in line[1:]]
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))

                if len(items) >= 3:
                    train_items = items[:-2]
                    length = min(len(train_items), self.maxlen)
                    for t in range(length):
                        self.trainData.append([train_items[:-length + t], train_items[-length + t]])
                    self.valData[user] = [items[:-2], items[-2]]
                    self.testData[user] = [items[:-2], items[-1]]
                else:
                    for t in range(len(items)):
                        self.trainData.append([items[:-len(items) + t], items[-len(items) + t]])
                    self.valData[user] = []
                    self.testData[user] = []

        with open(test_file, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                user, negatives = int(line[0]) - 1, [int(item) for item in line[1:]]
                self.allPos[user] = negatives

        self.n_user, self.m_item = self.n_user + 1, self.m_item + 1

    def __getitem__(self, idx):
        seq, label = self.trainData[idx]
        return seq, label

    def __len__(self):
        return len(self.trainData)

@dataclass
class SequentialCollator:
    def __call__(self, batch) -> dict:
        seqs, labels = zip(*batch)
        max_len = max(max([len(seq) for seq in seqs]), 2)
        inputs = [[0] * (max_len - len(seq)) + seq for seq in seqs]
        inputs_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs]
        labels = [[label] for label in labels]
        inputs, inputs_mask, labels = torch.LongTensor(inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)

        return {
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "labels": labels
        }
