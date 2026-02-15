"""
DrawRNN — Train and evaluate an LSTM sketch classifier.

Usage
-----
    python main.py                          # train with defaults
    python main.py --epochs 100 --lr 0.0005 # override hyper-params
    python main.py --test checkpoints/best_model.pt  # evaluate only
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from src.config import Config
from src.dataset import (
    build_label_maps,
    create_dataloaders,
    load_category,
    process_drawing,
)
from src.model import DrawRNNModel
from src.train import evaluate, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DrawRNN — Quick, Draw! sketch classifier")

    # data
    p.add_argument("--data-dir", type=str, default=None, help="Path to ndjson data folder")
    p.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Category names to train on (e.g. ant axe bat)",
    )

    # model
    p.add_argument("--hidden", type=int, default=None, help="LSTM hidden size")
    p.add_argument("--layers", type=int, default=None, help="Number of LSTM layers")
    p.add_argument("--dropout", type=float, default=None, help="Dropout between LSTM layers")

    # training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])

    # eval-only
    p.add_argument("--test", type=str, default=None, help="Path to checkpoint; run test only")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    # Override config with CLI args
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.categories:
        cfg.categories = args.categories
    if args.hidden:
        cfg.hidden_size = args.hidden
    if args.layers:
        cfg.num_layers = args.layers
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.lr:
        cfg.learning_rate = args.lr
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.patience:
        cfg.patience = args.patience
    if args.device:
        cfg.device = args.device

    # ── Label maps ────────────────────────────────────────────────────────
    encode, decode = build_label_maps(cfg.categories)
    print(f"Categories: {cfg.categories}")
    print(f"Label mapping: {encode}\n")

    # ── Load & process data ───────────────────────────────────────────────
    all_data: list[dict] = []
    for cat in cfg.categories:
        print(f"Loading '{cat}' …", end=" ")
        raw = load_category(cfg.data_dir, cat)
        print(f"{len(raw)} drawings")
        for d in raw:
            all_data.append(process_drawing(d, encode[cat]))

    random.shuffle(all_data)
    print(f"\nTotal drawings: {len(all_data)}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        all_data,
        batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model = DrawRNNModel(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        output_size=cfg.num_classes,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()

    # ── Test-only mode ────────────────────────────────────────────────────
    if args.test:
        ckpt = torch.load(args.test, map_location=cfg.device)
        model.load_state_dict(ckpt)
        test_loss, test_acc = evaluate(model, test_loader, criterion, cfg.device)
        print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")
        return

    # ── Train ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    model = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=cfg.num_epochs,
        device=cfg.device,
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        checkpoint_dir=cfg.checkpoint_dir,
    )

    # ── Evaluate on test set ──────────────────────────────────────────────
    test_loss, test_acc = evaluate(model, test_loader, criterion, cfg.device)
    print(f"\nTest Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
