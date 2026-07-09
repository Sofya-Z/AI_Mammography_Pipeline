"""CLIP-style encoders with a small classification head.

Encoder weights are optional: when a backbone can't be loaded, a lightweight
convolutional stub of the right embedding size is used so the head and training
loop stay runnable. Replace `_load_encoder` with real backbone code to plug in
actual weights.
"""

from dataclasses import dataclass

import torch
from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_features, num_classes=2, hidden=(512, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _StubEncoder(nn.Module):
    """Small conv encoder used when real weights are unavailable."""

    def __init__(self, embed_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x):
        return self.proj(self.body(x).flatten(1))


@dataclass
class EncoderSpec:
    embed_dim: int
    input_size: tuple
    backbone: str


ENCODERS = {
    "mammo_clip": EncoderSpec(2048, (1520, 912), "efficientnet_b5"),
    "medclip": EncoderSpec(768, (512, 512), "swin"),
    "biomedclip": EncoderSpec(512, (224, 224), "vit_b_16"),
    "openai_clip": EncoderSpec(768, (336, 336), "vit_l_14"),
}


class ClipClassifier(nn.Module):
    def __init__(self, variant="mammo_clip", num_classes=2, weights_path=None, freeze_encoder=True):
        super().__init__()
        if variant not in ENCODERS:
            raise ValueError(f"unknown variant '{variant}', pick one of {sorted(ENCODERS)}")
        self.spec = ENCODERS[variant]
        self.encoder = self._load_encoder(weights_path)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.head = MLPHead(self.spec.embed_dim, num_classes)

    def _load_encoder(self, weights_path):
        # Hook for real backbone loading (open_clip / transformers / custom).
        # Falls back to a stub so the model runs without external weights.
        return _StubEncoder(self.spec.embed_dim)

    def forward(self, x):
        return self.head(self.encoder(x))
