"""Config loading, losses and metrics."""

import importlib

from .config import apply_overrides, load_config, read_yaml

_LAZY = {
    "FocalLoss": "training",
    "make_loss": "training",
    "balanced_sampler": "training",
    "auroc": "training",
    "f1_score": "training",
}

__all__ = ["apply_overrides", "load_config", "read_yaml", *_LAZY]


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
