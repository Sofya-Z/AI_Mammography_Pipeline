#!/usr/bin/env python
"""Train and evaluate a model from a YAML config."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mammokit.train import run
from mammokit.utils.config import apply_overrides, load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="e.g. train.lr=0.0005 transforms.preset=full")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    result = run(cfg)
    print("\nsummary:", json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
