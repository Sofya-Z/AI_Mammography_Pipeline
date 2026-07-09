"""Torch dataset that reads mammograms from a CSV of annotations."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from .dicom_reader import read_mammogram_kw


class MammogramDataset(Dataset):
    """One row per image; yields (image_tensor, label)."""

    def __init__(
        self,
        ann_file,
        root,
        transforms=None,
        reader=None,
        read_options=None,
        image_col="image",
        label_col="label",
        source=None,
        class_map=None,
        drop_classes=None,
    ):
        self.root = Path(root)
        self.transforms = transforms
        self.reader = reader or read_mammogram_kw
        self.read_options = read_options or {}
        self.image_col = image_col
        self.label_col = label_col
        self.source = source

        table = pd.read_csv(ann_file)
        if drop_classes:
            table = table[~table[label_col].isin(drop_classes)]
        if class_map:
            table = table[table[label_col].isin(class_map)]
            table[label_col] = table[label_col].map(class_map)

        self.table = table.reset_index(drop=True)
        self.classes = sorted(self.table[label_col].unique())

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        img = self.reader(str(self.root / row[self.image_col]), read_options=self.read_options)

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        else:
            img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1)

        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return img, label


def concat_sources(specs, split, transforms=None):
    """Build one concatenated dataset from several source specs for a given split."""
    parts = []
    for spec in specs:
        ann = spec.get("ann_files", {}).get(split)
        if not ann:
            continue
        parts.append(
            MammogramDataset(
                ann_file=ann,
                root=spec["root"],
                transforms=transforms,
                read_options=spec.get("read_options"),
                source=spec.get("name"),
                class_map=spec.get("class_map"),
                drop_classes=spec.get("drop_classes"),
            )
        )
    if not parts:
        raise ValueError(f"no source has a '{split}' split")
    return ConcatDataset(parts)
