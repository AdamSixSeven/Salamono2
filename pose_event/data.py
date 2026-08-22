from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from .features import augment_features


class WindowDataset(Dataset):
    def __init__(self, x, y_action, y_safety, mask, *, augment=False, seed=42):
        self.x = np.asarray(x, dtype=np.float32)
        self.y_action = np.asarray(y_action, dtype=np.int64)
        self.y_safety = np.asarray(y_safety, dtype=np.int64)
        self.mask = np.asarray(mask, dtype=np.bool_)
        self.augment = augment
        self.seed = int(seed)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.augment:
            rng = np.random.default_rng(self.seed + idx + np.random.randint(0, 1_000_000))
            x = augment_features(x, rng)
        return (
            torch.from_numpy(x),
            torch.tensor(self.y_action[idx]),
            torch.tensor(self.y_safety[idx]),
            torch.from_numpy(self.mask[idx]),
        )


def load_window_npz(path):
    data = np.load(path, allow_pickle=False)
    required = {"X", "y_action", "y_safety", "mask", "split"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Missing arrays in dataset: {sorted(missing)}")
    return {key: data[key] for key in data.files}
