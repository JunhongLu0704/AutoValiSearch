from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import TinyVisionNet
from .stable_reweighting import stable_sample_weights
from .synthetic_domain import build_synthetic_loaders


REQUIRED_CONFIG_FIELDS = ("lr", "lambdap", "epochp", "num_f")


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_CONFIG_FIELDS if name not in config]
    if missing:
        raise ValueError(f"missing required config fields: {missing}")
    return {
        "lr": float(config["lr"]),
        "lambdap": float(config["lambdap"]),
        "epochp": int(config["epochp"]),
        "num_f": int(config["num_f"]),
        "epochs": int(config.get("epochs", 4)),
        "batch_size": int(config.get("batch_size", 64)),
        "seed": int(config.get("seed", 0)),
        "samples_per_split": int(config.get("samples_per_split", 160)),
    }


def run_training_trial(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Run a small but real PyTorch vision training trial.

    The function mirrors the artifact shape of the full research pipeline:
    it writes epoch checkpoints plus a `result.json` summary. It uses a
    synthetic domain-shift dataset so the public repo remains runnable without
    private data.
    """

    cfg = normalize_config(config)
    out = Path(output_dir)
    ckpt_dir = out / "epoch_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_synthetic_loaders(
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        samples_per_split=cfg["samples_per_split"],
    )
    model = TinyVisionNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none")

    history = []
    best_val = -1.0
    best_epoch = -1
    for epoch in range(cfg["epochs"]):
        train_metrics = _train_one_epoch(model, train_loader, optimizer, criterion, cfg, epoch, device)
        val_metrics = _evaluate(model, val_loader, device)
        test_metrics = _evaluate(model, test_loader, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}, **{f"test_{k}": v for k, v in test_metrics.items()}}
        history.append(row)
        torch.save(
            {"epoch": epoch, "config": cfg, "model_state": model.state_dict(), "metrics": row},
            ckpt_dir / f"epoch_{epoch:03d}.pt",
        )
        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            best_epoch = epoch

    selected = history[best_epoch]
    result = {
        "status": "ok",
        "config": cfg,
        "best_epoch_by_val": best_epoch,
        "best_val_acc": round(float(best_val), 6),
        "selected_test_acc": round(float(selected["test_acc"]), 6),
        "history": history,
        "artifact_note": "synthetic public training trial; not a full PACS/VLCS formal run",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _train_one_epoch(model, loader, optimizer, criterion, cfg, epoch, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, targets, _domains in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits, features = model(images, return_features=True)
        per_sample_loss = criterion(logits, targets)
        if epoch >= cfg["epochp"]:
            weights = stable_sample_weights(features, cfg["lambdap"], cfg["num_f"]).to(device)
            loss = (per_sample_loss * weights).mean()
        else:
            loss = per_sample_loss.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * targets.numel()
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total += targets.numel()
    return {"loss": total_loss / max(total, 1), "acc": 100.0 * total_correct / max(total, 1)}


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total = 0
    for images, targets, _domains in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total += targets.numel()
    return {"acc": 100.0 * total_correct / max(total, 1)}
