#!/usr/bin/env python
"""Create a stratified train/validation/test split from a CSV."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mammokit.data.splits import save_splits, split_dataset, split_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col", default="dataset")
    ap.add_argument("--valid-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.ann)
    group_col = args.group_col if args.group_col in df.columns else None

    splits = split_dataset(
        df,
        label_col=args.label_col,
        group_col=group_col,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    paths = save_splits(splits, args.out_dir)
    print(split_summary(splits, label_col=args.label_col).to_string(index=False))
    for name, p in paths.items():
        print(f"{name}: {p}")


if __name__ == "__main__":
    main()
