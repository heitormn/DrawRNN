"""
Data loading and dataset utilities for Quick, Draw! ndjson files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Label encoding helpers
# ---------------------------------------------------------------------------

def build_label_maps(categories: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return (encode, decode) dictionaries for a list of category names."""
    encode = {name: idx for idx, name in enumerate(sorted(categories))}
    decode = {idx: name for name, idx in encode.items()}
    return encode, decode


# ---------------------------------------------------------------------------
# Raw I/O
# ---------------------------------------------------------------------------

def load_ndjson(filepath: str | Path) -> List[dict]:
    """Load a Quick, Draw! simplified ndjson file and return a list of drawing dicts."""
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        drawings = [json.loads(line) for line in f]
    return drawings


def load_category(data_dir: str | Path, category: str) -> List[dict]:
    """Convenience: load ``data_dir/full_simplified_<category>.ndjson``."""
    path = Path(data_dir) / f"full_simplified_{category}.ndjson"
    return load_ndjson(path)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_drawing(drawing_json: dict, label: int) -> dict:
    """
    Convert a drawing JSON dict into a dict with:
      - "label"       : int tensor
      - "coordinates"  : float tensor of shape (num_points, 3)  →  [x, y, stroke_progress]
      - "word"         : original class name
    """
    drawing = drawing_json["drawing"]
    all_points: list = []

    for stroke in drawing:
        x_coords = np.array(stroke[0], dtype=np.float32)
        y_coords = np.array(stroke[1], dtype=np.float32)
        num_points = len(x_coords)
        progress = np.linspace(0, 1, num_points, dtype=np.float32)

        for i in range(num_points):
            all_points.append([x_coords[i], y_coords[i], progress[i]])

    return {
        "label": torch.tensor(label, dtype=torch.long),
        "coordinates": torch.tensor(all_points, dtype=torch.float32),
        "word": drawing_json.get("word", "unknown"),
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class DrawingDataset(Dataset):
    """Simple map-style dataset wrapping processed drawing dicts."""

    def __init__(self, data: List[dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]["coordinates"], self.data[idx]["label"]


# ---------------------------------------------------------------------------
# Collate & DataLoader factory
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Pad variable-length coordinate sequences to the longest in the batch."""
    coords, labels = zip(*batch)
    coords_padded = pad_sequence(coords, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return coords_padded, labels


def create_dataloaders(
    data: List[dict],
    batch_size: int = 256,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split *data* into train / val / test and return three DataLoaders.

    Parameters
    ----------
    data : list of processed drawing dicts
    batch_size : mini-batch size
    train_ratio, val_ratio : fractions; test gets the remainder
    num_workers : DataLoader workers

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    total = len(data)
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_ds = DrawingDataset(data[:train_end])
    val_ds = DrawingDataset(data[train_end:val_end])
    test_ds = DrawingDataset(data[val_end:])

    common = dict(batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader
