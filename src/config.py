"""
Configuration defaults for DrawRNN.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """All hyper-parameters and paths in one place."""

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "data/Drawings"
    categories: List[str] = field(default_factory=lambda: ["ant", "axe", "bat"])

    # ── Splits ────────────────────────────────────────────────────────────
    train_ratio: float = 0.8
    val_ratio: float = 0.1  # test = 1 - train - val

    # ── Model ─────────────────────────────────────────────────────────────
    input_size: int = 3        # x, y, stroke_progress
    hidden_size: int = 128
    num_layers: int = 4
    dropout: float = 0.0

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 256
    learning_rate: float = 1e-3
    num_epochs: int = 50
    patience: int = 5
    min_delta: float = 0.001
    device: str = "cpu"

    # ── Outputs ───────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"

    @property
    def num_classes(self) -> int:
        return len(self.categories)
