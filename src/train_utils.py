from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


CRITERION_CE = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
        total_n += x.size(0)

    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
    }


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
        total_n += x.size(0)

    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
    }


def train_model(
    model: nn.Module,
    dl_train,
    dl_val,
    dl_test,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, list[float]]:
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    history: Dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, dl_train, CRITERION_CE, optimizer, device
        )
        val_metrics = evaluate_epoch(
            model, dl_val, CRITERION_CE, device
        )
        test_metrics = evaluate_epoch(
            model, dl_test, CRITERION_CE, device
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train: acc={train_metrics['acc']:.3f}, loss={train_metrics['loss']:.3f} | "
            f"val: acc={val_metrics['acc']:.3f}, loss={val_metrics['loss']:.3f} | "
            f"test: acc={test_metrics['acc']:.3f}, loss={test_metrics['loss']:.3f}"
        )

    return history


def merge_histories(
    h1: Dict[str, list[float]],
    h2: Dict[str, list[float]]
) -> Dict[str, list[float]]:
    return {k: list(h1[k]) + list(h2[k]) for k in h1}
