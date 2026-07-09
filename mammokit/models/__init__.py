"""Model factory."""

from .clip_models import ENCODERS, ClipClassifier, MLPHead
from .efficientnet import EfficientNetClassifier


def make_model(cfg):
    name = cfg.get("name", "efficientnet").lower()
    num_classes = cfg.get("num_classes", 2)

    if name in ("efficientnet", "efficientnet_b3"):
        return EfficientNetClassifier(
            backbone=cfg.get("backbone", "efficientnet_b3"),
            weights=cfg.get("weights", "EfficientNet_B3_Weights.DEFAULT"),
            num_classes=num_classes,
        )

    if name in ENCODERS:
        return ClipClassifier(
            variant=name,
            num_classes=num_classes,
            weights_path=cfg.get("weights_path"),
            freeze_encoder=cfg.get("freeze_encoder", True),
        )

    raise ValueError(f"unknown model '{name}'")


__all__ = ["EfficientNetClassifier", "ClipClassifier", "ENCODERS", "MLPHead", "make_model"]
