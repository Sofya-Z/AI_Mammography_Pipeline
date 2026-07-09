"""Stratified train/validation/test splitting."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _key(df, label_col, group_col):
    key = df[label_col].astype(str)
    if group_col and group_col in df.columns:
        key = df[group_col].astype(str) + "::" + key
    return key


def split_dataset(df, label_col="label", group_col="dataset", valid_size=0.15, test_size=0.15, seed=42):
    """Split into train/valid/test, stratified by label (and group if present).

    Falls back to a random split for any stage where a stratum is too small.
    """
    if not 0 < test_size < 1 or not 0 <= valid_size < 1 or valid_size + test_size >= 1:
        raise ValueError("valid_size and test_size must be in [0, 1) and sum to < 1")

    df = df.reset_index(drop=True)
    key = _key(df, label_col, group_col)
    strat = key if key.value_counts().min() >= 2 else None

    trainval, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)

    rel_valid = valid_size / (1.0 - test_size)
    key_tv = _key(trainval, label_col, group_col)
    strat_tv = key_tv if key_tv.value_counts().min() >= 2 else None

    train, valid = train_test_split(trainval, test_size=rel_valid, random_state=seed, stratify=strat_tv)

    return {
        "train": train.reset_index(drop=True),
        "valid": valid.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def save_splits(splits, out_dir, prefix=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in splits.items():
        p = out_dir / f"{prefix}{name}.csv"
        frame.to_csv(p, index=False)
        paths[name] = p
    return paths


def split_summary(splits, label_col="label"):
    rows = []
    for name, frame in splits.items():
        pos = float((frame[label_col] == 1).mean()) if len(frame) else 0.0
        rows.append({"split": name, "n": len(frame), "positive_fraction": round(pos, 4)})
    return pd.DataFrame(rows)
