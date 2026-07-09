"""Training and evaluation loop."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.augmentations import make_pipeline_from_config
from .data.dataset import concat_sources
from .models import make_model
from .utils.training import auroc, f1_score, make_loss


def _source_specs(cfg):
    shared = cfg.get("source_defaults", {})
    specs = []
    for spec in cfg["sources"]:
        merged = {**shared, **spec}
        merged.setdefault("read_options", shared.get("read_options"))
        specs.append(merged)
    return specs


def build_loaders(cfg):
    specs = _source_specs(cfg)
    tcfg = cfg.get("transforms", {})
    train_tf = make_pipeline_from_config(tcfg, train=True)
    eval_tf = make_pipeline_from_config(tcfg, train=False)

    bs = cfg["train"].get("batch_size", 8)
    nw = cfg["train"].get("num_workers", 4)

    def collate(batch):
        imgs, labels = zip(*batch)
        return torch.stack(imgs), torch.stack(labels)

    loaders = {}
    for split in ("train", "valid", "test"):
        try:
            ds = concat_sources(specs, split, transforms=train_tf if split == "train" else eval_tf)
        except ValueError:
            continue
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=(split == "train"), num_workers=nw, collate_fn=collate
        )
    return loaders


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for imgs, y in loader:
        p = torch.softmax(model(imgs.to(device)), dim=1)[:, 1]
        scores.append(p.cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    return {"auroc": auroc(s, y), "f1": f1_score(s, y)}


def run(cfg):
    """Train, pick the best epoch by validation AUROC, evaluate on test."""
    device = torch.device(cfg["train"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["train"].get("seed", 42))

    loaders = build_loaders(cfg)
    model = make_model(cfg["model"]).to(device)
    criterion = make_loss(cfg["train"].get("loss", {}))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg["train"].get("lr", 1e-4)
    )

    epochs = cfg["train"].get("num_epochs", 30)
    best_auroc, best_state, history = -1.0, None, []

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for imgs, y in loaders["train"]:
            imgs, y = imgs.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), y)
            loss.backward()
            optimizer.step()
            total += loss.item()

        row = {"epoch": epoch, "train_loss": total / max(len(loaders["train"]), 1)}
        if "valid" in loaders:
            val = evaluate(model, loaders["valid"], device)
            row.update({f"valid_{k}": v for k, v in val.items()})
            if not np.isnan(val["auroc"]) and val["auroc"] > best_auroc:
                best_auroc = val["auroc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        print(f"epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))

    if best_state is not None:
        model.load_state_dict(best_state)

    result = {"best_valid_auroc": best_auroc, "history": history}
    if "test" in loaders:
        result["test"] = evaluate(model, loaders["test"], device)
        print(f"test: auroc={result['test']['auroc']:.4f} f1={result['test']['f1']:.4f}")

    ckpt = cfg["train"].get("checkpoint")
    if ckpt:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)

    return result
