"""YAML config loading with a simple include + override mechanism."""

from copy import deepcopy
from pathlib import Path

import yaml


def _merge(base, over):
    out = deepcopy(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def read_yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path, root=None):
    """Load a config file, resolving a top-level `includes:` list first."""
    path = Path(path)
    root = Path(root) if root else path.parent

    raw = read_yaml(path)
    includes = raw.pop("includes", []) or []

    merged = {}
    for inc in includes:
        merged = _merge(merged, read_yaml(root / inc))
    return _merge(merged, raw)


def apply_overrides(cfg, overrides):
    """Apply `a.b.c=value` overrides; values parsed as YAML scalars."""
    cfg = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key.subkey=value")
        dotted, value = item.split("=", 1)
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(value)
    return cfg
