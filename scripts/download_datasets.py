#!/usr/bin/env python
"""List sources or download a public mammography dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mammokit.data.download import fetch, show_access


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", help="dataset name or alias (vindr, cbis, cmmd, inbreast, mosmed)")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--info", action="store_true", help="print access instructions")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.info or not args.dataset:
        show_access([args.dataset] if args.dataset else None)
        if not args.dataset:
            return

    try:
        path = fetch(args.dataset, args.out, overwrite=args.overwrite)
        print(f"saved to {path}")
    except RuntimeError as exc:
        print(exc)


if __name__ == "__main__":
    main()
