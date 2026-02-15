"""
Training and evaluation loops for DrawRNN.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str | torch.device = "cpu",
    patience: int = 5,
    min_delta: float = 0.001,
    checkpoint_dir: Optional[str | Path] = None,
) -> nn.Module:
    """
    Train *model* with early stopping and optional checkpoint saving.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoaders
    criterion : loss function
    optimizer : optimizer
    num_epochs : maximum epochs
    device : "cpu" or "cuda"
    patience : early-stopping patience (epochs)
    min_delta : minimum improvement to reset patience counter
    checkpoint_dir : if given, save best model to this directory

    Returns
    -------
    model with best weights restored
    """
    model.to(device)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state: dict | None = None
    epoch = 0

    try:
        epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs", unit="epoch")
        for epoch in epoch_bar:
            # ---- training ----
            model.train()
            running_loss = 0.0
            batch_bar = tqdm(
                train_loader,
                desc=f"Train {epoch}/{num_epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for coords, labels in batch_bar:
                coords, labels = coords.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(coords)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * coords.size(0)
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            train_loss = running_loss / len(train_loader.dataset)

            # ---- validation ----
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")

            print(
                f"Epoch [{epoch}/{num_epochs}]  "
                f"Train Loss: {train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}  "
                f"Val Acc: {val_acc:.4f}"
            )

            # ---- early stopping ----
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = copy.deepcopy(model.state_dict())
                print(f"  -> Val loss improved to {best_val_loss:.4f}")
                if checkpoint_dir:
                    ckpt_path = Path(checkpoint_dir) / "best_model.pt"
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, ckpt_path)
            else:
                epochs_no_improve += 1
                print(f"  -> No improvement for {epochs_no_improve} epoch(s)")
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping after {epoch} epochs (best val loss: {best_val_loss:.4f})")
                    break

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}/{num_epochs}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Best model weights restored.")
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device = "cpu",
) -> tuple[float, float]:
    """Return (loss, accuracy) over the given DataLoader."""
    model.eval()
    model.to(device)
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for coords, labels in loader:
            coords, labels = coords.to(device), labels.to(device)
            outputs = model(coords)
            total_loss += criterion(outputs, labels).item() * coords.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n
