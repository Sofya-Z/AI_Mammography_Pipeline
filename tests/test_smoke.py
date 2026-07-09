"""Run the pipeline end to end on synthetic data."""

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pytest
except ImportError:
    pytest = None


def _has(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def _skip(reason):
    if pytest is not None:
        pytest.skip(reason)
    raise RuntimeError(f"SKIP: {reason}")


def test_filter_and_split():
    from mammokit.data.filtering import drop_metadata_anomalies
    from mammokit.data.splits import split_dataset

    df = pd.DataFrame({
        "image": [f"{i}.dcm" for i in range(40)],
        "label": [0, 1] * 20,
        "birads": [1, 5] * 20,
        "density": [2, 3] * 20,
        "dataset": ["synthetic"] * 40,
    })
    kept, removed = drop_metadata_anomalies(df)
    assert len(kept) == 40 and len(removed) == 0

    splits = split_dataset(df, valid_size=0.2, test_size=0.2, seed=0)
    assert sum(len(v) for v in splits.values()) == 40


def test_end_to_end():
    for mod in ("torch", "albumentations", "pydicom"):
        if not _has(mod):
            _skip(f"{mod} not installed")

    from make_synthetic_data import make_synthetic

    from mammokit.data.filtering import clean
    from mammokit.data.splits import save_splits, split_dataset
    from mammokit.train import run

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df = pd.read_csv(make_synthetic(tmp, per_class=8))
        kept, _ = clean(df, use_intensity=False, use_metadata=True)
        save_splits(split_dataset(kept, valid_size=0.25, test_size=0.25, seed=0), tmp)

        cfg = {
            "source_defaults": {"read_options": {"force_monochrome2": True}},
            "sources": [{
                "name": "synthetic",
                "root": str(tmp),
                "ann_files": {
                    "train": str(tmp / "train.csv"),
                    "valid": str(tmp / "valid.csv"),
                    "test": str(tmp / "test.csv"),
                },
                "class_map": {0: 0, 1: 1},
            }],
            "transforms": {
                "preset": "standard",
                "resolution": {"height": 64, "width": 64, "keep_ratio": False},
                "normalize": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
            },
            "model": {"name": "efficientnet", "num_classes": 2, "weights": None},
            "train": {
                "seed": 0, "device": "cpu", "num_epochs": 2,
                "batch_size": 4, "num_workers": 0, "lr": 1e-3,
                "loss": {"type": "FocalLoss", "kwargs": {"alpha": 0.5, "gamma": 2.0}},
            },
        }
        result = run(cfg)
        assert len(result["history"]) == 2 and "test" in result


if __name__ == "__main__":
    test_filter_and_split()
    print("filter + split OK")
    try:
        test_end_to_end()
        print("end-to-end OK")
    except RuntimeError as exc:
        if str(exc).startswith("SKIP:"):
            print(f"end-to-end skipped ({str(exc)[6:].strip()})")
        else:
            raise
