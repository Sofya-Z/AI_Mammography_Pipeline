"""Image augmentation presets built on Albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# name -> Albumentations class
TRANSFORMS = {
    "Resize": A.Resize,
    "LongestMaxSize": A.LongestMaxSize,
    "PadIfNeeded": A.PadIfNeeded,
    "Affine": A.Affine,
    "HorizontalFlip": A.HorizontalFlip,
    "VerticalFlip": A.VerticalFlip,
    "RandomRotate90": A.RandomRotate90,
    "GridDistortion": A.GridDistortion,
    "OpticalDistortion": A.OpticalDistortion,
    "CLAHE": A.CLAHE,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "RandomGamma": A.RandomGamma,
    "RandomToneCurve": A.RandomToneCurve,
    "HueSaturationValue": A.HueSaturationValue,
    "Sharpen": A.Sharpen,
    "AdvancedBlur": A.AdvancedBlur,
    "GaussNoise": A.GaussNoise,
    "PixelDropout": A.PixelDropout,
    "CoarseDropout": A.CoarseDropout,
    "GridDropout": A.GridDropout,
}

# default kwargs per transform
PARAMS = {
    "Affine": {
        "p": 0.35,
        "translate_percent": [-0.0625, 0.0625],
        "scale": [0.9, 1.1],
        "rotate": [-45, 45],
        "shear": 0,
        "border_mode": 0,
        "fill": 0,
        "keep_ratio": False,
    },
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "RandomRotate90": {"p": 0.5},
    "GridDistortion": {"p": 0.35, "border_mode": 0},
    "OpticalDistortion": {
        "p": 0.5,
        "distort_limit": [-0.3, 0.3],
        "interpolation": 1,
        "mode": "camera",
        "border_mode": 0,
    },
    "CLAHE": {"clip_limit": 1.5, "tile_grid_size": [8, 8], "p": 0.35},
    "RandomBrightnessContrast": {
        "brightness_limit": [-0.1, 0.1],
        "contrast_limit": [-0.2, 0.2],
        "brightness_by_max": True,
        "p": 0.5,
    },
    "RandomGamma": {"gamma_limit": [80, 120], "p": 0.5},
    "RandomToneCurve": {"scale": 0.1, "p": 0.5},
    "HueSaturationValue": {"hue_shift_limit": 0, "sat_shift_limit": 0, "val_shift_limit": 20, "p": 0.5},
    "Sharpen": {"p": 0.35, "alpha": [0.02, 0.2]},
    "AdvancedBlur": {"p": 0.35, "blur_limit": [3, 5]},
    "GaussNoise": {"p": 0.5, "std_range": [0.01, 0.05]},
    "PixelDropout": {"p": 0.35},
    "CoarseDropout": {"p": 0.35},
    "GridDropout": {"p": 0.35, "ratio": 0.35},
}

# named presets
PRESETS = {
    "light": [
        "HorizontalFlip",
        "VerticalFlip",
        "HueSaturationValue",
        "Sharpen",
        "RandomBrightnessContrast",
    ],
    "standard": [
        "Affine",
        "RandomRotate90",
        "HorizontalFlip",
        "VerticalFlip",
        "GridDistortion",
        "OpticalDistortion",
        "CLAHE",
        "RandomBrightnessContrast",
        "RandomGamma",
        "Sharpen",
        "PixelDropout",
        "GridDropout",
    ],
    "full": list(PARAMS.keys()),
    "none": [],
}


def _resize_ops(height, width, keep_ratio):
    if keep_ratio:
        return [
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(min_height=height, min_width=width, border_mode=0),
        ]
    return [A.Resize(height=height, width=width)]


def make_pipeline(
    preset="standard",
    height=1024,
    width=1024,
    keep_ratio=False,
    mean=(0.1553, 0.1553, 0.1553),
    std=(0.2344, 0.2344, 0.2344),
    train=True,
    params=None,
):
    """Compose an augmentation + normalization pipeline for a named preset."""
    preset = preset.lower()
    if preset not in PRESETS:
        raise ValueError(f"unknown preset '{preset}', pick one of {sorted(PRESETS)}")

    merged = {k: dict(v) for k, v in PARAMS.items()}
    for name, kw in (params or {}).items():
        merged.setdefault(name, {}).update(kw)

    ops = _resize_ops(height, width, keep_ratio)
    if train:
        for name in PRESETS[preset]:
            ops.append(TRANSFORMS[name](**merged.get(name, {})))
    ops.append(A.Normalize(mean=list(mean), std=list(std), max_pixel_value=1.0))
    ops.append(ToTensorV2())
    return A.Compose(ops)


def make_pipeline_from_config(cfg, train=True):
    res = cfg.get("resolution", {})
    norm = cfg.get("normalize", {})
    return make_pipeline(
        preset=cfg.get("preset", "standard"),
        height=res.get("height", 1024),
        width=res.get("width", 1024),
        keep_ratio=res.get("keep_ratio", False),
        mean=tuple(norm.get("mean", (0.1553, 0.1553, 0.1553))),
        std=tuple(norm.get("std", (0.2344, 0.2344, 0.2344))),
        train=train,
        params=cfg.get("params"),
    )
