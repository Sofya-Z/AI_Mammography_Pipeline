"""Reading and normalizing mammography DICOM files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
except ImportError:
    pydicom = None
    apply_modality_lut = apply_voi_lut = None


@dataclass
class ReadOptions:
    apply_modality_lut: bool = False
    apply_voi_lut: bool = False
    force_monochrome2: bool = True
    clip: bool = False
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid: tuple = (8, 8)


def _open(path):
    if pydicom is None:
        raise ImportError("pydicom is not installed (pip install pydicom)")
    return pydicom.dcmread(str(path))


def _equalize(gray_u8, clip_limit, grid):
    import cv2

    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(grid)).apply(gray_u8)


def read_mammogram(path, options=None):
    """Load a DICOM and return an (H, W, 3) float32 array scaled to [0, 1].

    MONOCHROME1 images are inverted so that background is dark and dense tissue
    is bright, which puts every source on the same convention.
    """
    opt = options or ReadOptions()
    ds = _open(path)
    pixels = ds.pixel_array.astype(np.float32)

    if opt.apply_modality_lut and apply_modality_lut is not None:
        pixels = apply_modality_lut(pixels, ds).astype(np.float32)
    if opt.apply_voi_lut and apply_voi_lut is not None:
        pixels = apply_voi_lut(pixels, ds).astype(np.float32)

    if opt.force_monochrome2:
        interp = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
        if interp == "MONOCHROME1":
            pixels = pixels.max() - pixels

    span = float(pixels.max() - pixels.min())
    pixels = (pixels - pixels.min()) / span if span > 0 else np.zeros_like(pixels)

    if opt.clip:
        pixels = np.clip(pixels, 0.0, 1.0)

    if opt.clahe:
        eq = _equalize((pixels * 255).astype(np.uint8), opt.clahe_clip_limit, opt.clahe_grid)
        pixels = eq.astype(np.float32) / 255.0

    return np.repeat(pixels[..., None], 3, axis=-1).astype(np.float32)


def read_mammogram_kw(path, read_options=None, **_):
    """Config-friendly wrapper: accepts a plain dict of read options."""
    return read_mammogram(path, ReadOptions(**(read_options or {})))
