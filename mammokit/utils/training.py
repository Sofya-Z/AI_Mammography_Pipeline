"""Loss functions, samplers and metrics."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_loss(cfg):
    kind = cfg.get("type", "FocalLoss")
    kwargs = cfg.get("kwargs", {})
    if kind == "FocalLoss":
        return FocalLoss(**kwargs)
    if kind == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(**kwargs)
    raise ValueError(f"unknown loss '{kind}'")


def balanced_sampler(labels, groups=None, num_samples=None):
    """WeightedRandomSampler that oversamples rare (group, label) strata."""
    from torch.utils.data import WeightedRandomSampler

    labels = np.asarray(labels)
    if groups is None:
        strata = labels
    else:
        groups = np.asarray(groups)
        strata = np.array([f"{g}::{y}" for g, y in zip(groups, labels)])

    _, inv, counts = np.unique(strata, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inv]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples or len(labels),
        replacement=True,
    )


def auroc(scores, labels):
    """AUROC via average ranks; returns nan if a class is absent."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank = (cum - counts + cum + 1) / 2.0
    ranks = avg_rank[inv]

    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def f1_score(scores, labels, threshold=0.5):
    preds = (np.asarray(scores) >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0
