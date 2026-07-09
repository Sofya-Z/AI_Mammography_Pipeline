"""Dropping bad samples by pixel statistics and metadata consistency."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def mean_intensity(df, root, reader, image_col="image", ignore_background=True):
    """Mean pixel intensity per image, optionally excluding zero background."""
    root = Path(root)
    out = []
    for rel in df[image_col]:
        arr = np.asarray(reader(str(root / rel)), dtype=np.float32)
        if ignore_background:
            fg = arr[arr > 0]
            out.append(float(fg.mean()) if fg.size else 0.0)
        else:
            out.append(float(arr.mean()))
    return pd.Series(out, index=df.index, name="mean_intensity")


def drop_intensity_tails(df, low_pct=5.0, high_pct=5.0, col="mean_intensity"):
    """Remove the darkest low_pct% and brightest high_pct% images."""
    if col not in df.columns:
        raise KeyError(f"column '{col}' missing; compute mean_intensity first")

    vals = df[col].to_numpy()
    lo = np.percentile(vals, low_pct) if low_pct > 0 else -np.inf
    hi = np.percentile(vals, 100 - high_pct) if high_pct > 0 else np.inf
    keep = (df[col] >= lo) & (df[col] <= hi)
    return df[keep].copy(), df[~keep].copy()


@dataclass
class MetadataRules:
    valid_density: set = field(default_factory=lambda: {1, 2, 3, 4})
    valid_birads: set = field(default_factory=lambda: {1, 2, 5})
    label_col: str = "label"
    birads_col: str = "birads"
    density_col: str = "density"
    positive_label: int = 1
    positive_birads: set = field(default_factory=lambda: {4, 5, 6})


def _as_int(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def drop_metadata_anomalies(df, rules=None):
    """Remove rows with out-of-range or self-contradictory metadata."""
    rules = rules or MetadataRules()
    bad = pd.Series(False, index=df.index)

    if rules.density_col in df.columns:
        d = _as_int(df[rules.density_col])
        bad |= d.notna() & ~d.isin(rules.valid_density)

    if rules.birads_col in df.columns:
        b = _as_int(df[rules.birads_col])
        bad |= b.notna() & ~b.isin(rules.valid_birads)

    # a positive label with a clearly benign BI-RADS is contradictory
    if rules.label_col in df.columns and rules.birads_col in df.columns:
        b = _as_int(df[rules.birads_col])
        positive = df[rules.label_col] == rules.positive_label
        bad |= positive & b.notna() & ~b.isin(rules.positive_birads)

    return df[~bad].copy(), df[bad].copy()


def clean(
    df,
    root=None,
    reader=None,
    low_pct=5.0,
    high_pct=5.0,
    rules=None,
    use_intensity=True,
    use_metadata=True,
    image_col="image",
):
    """Run metadata then intensity filtering; return (kept, {stage: removed})."""
    removed = {}
    kept = df

    if use_metadata:
        kept, removed["metadata"] = drop_metadata_anomalies(kept, rules)

    if use_intensity:
        if "mean_intensity" not in kept.columns:
            if reader is None or root is None:
                raise ValueError("intensity filtering needs reader + root, or a precomputed column")
            kept = kept.copy()
            kept["mean_intensity"] = mean_intensity(kept, root, reader, image_col=image_col)
        kept, removed["intensity"] = drop_intensity_tails(kept, low_pct, high_pct)

    return kept.reset_index(drop=True), removed
