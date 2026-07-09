#!/usr/bin/env python
"""Filter a dataset CSV by metadata consistency and pixel-intensity tails."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mammokit.data.dicom_reader import read_mammogram_kw
from mammokit.data.filtering import MetadataRules, clean


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann", required=True, help="input annotation CSV")
    ap.add_argument("--root", help="image root (needed if intensity not precomputed)")
    ap.add_argument("--out", required=True, help="output filtered CSV")
    ap.add_argument("--low-pct", type=float, default=5.0)
    ap.add_argument("--high-pct", type=float, default=5.0)
    ap.add_argument("--no-intensity", action="store_true")
    ap.add_argument("--no-metadata", action="store_true")
    ap.add_argument("--report", help="optional CSV of removed rows")
    args = ap.parse_args()

    df = pd.read_csv(args.ann)
    print(f"loaded {len(df)} rows")

    reader = None
    if not args.no_intensity and "mean_intensity" not in df.columns:
        reader = read_mammogram_kw

    kept, removed = clean(
        df,
        root=args.root,
        reader=reader,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
        rules=MetadataRules(),
        use_intensity=not args.no_intensity,
        use_metadata=not args.no_metadata,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(args.out, index=False)
    print(f"kept {len(kept)} (metadata -{len(removed.get('metadata', []))}, "
          f"intensity -{len(removed.get('intensity', []))}) -> {args.out}")

    if args.report and any(len(f) for f in removed.values()):
        merged = pd.concat(
            [f.assign(removed_by=k) for k, f in removed.items() if len(f)], ignore_index=True
        )
        merged.to_csv(args.report, index=False)
        print(f"removal report -> {args.report}")


if __name__ == "__main__":
    main()
