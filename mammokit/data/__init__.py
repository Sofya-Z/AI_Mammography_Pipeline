"""Data handling: download, reading, filtering, augmentation, splitting.

Heavy dependencies (torch, albumentations) are imported lazily so the light
stages work without a full deep-learning stack installed.
"""

import importlib

_EXPORTS = {
    "make_pipeline": "augmentations",
    "make_pipeline_from_config": "augmentations",
    "PRESETS": "augmentations",
    "MammogramDataset": "dataset",
    "concat_sources": "dataset",
    "ReadOptions": "dicom_reader",
    "read_mammogram": "dicom_reader",
    "read_mammogram_kw": "dicom_reader",
    "MetadataRules": "filtering",
    "drop_intensity_tails": "filtering",
    "drop_metadata_anomalies": "filtering",
    "mean_intensity": "filtering",
    "clean": "filtering",
    "split_dataset": "splits",
    "split_summary": "splits",
    "save_splits": "splits",
    "fetch": "download",
    "find": "download",
    "show_access": "download",
    "SOURCES": "download",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
