"""
Visualization utilities for Quick, Draw! sketches.
"""

from __future__ import annotations

import time
from typing import Optional

import matplotlib.pyplot as plt


def draw_sketch(
    drawing_json: dict,
    delay: float = 0.1,
    animated: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Render a Quick, Draw! drawing.

    Parameters
    ----------
    drawing_json : dict with at least a "drawing" key (list of strokes)
    delay : seconds between strokes when *animated* is True
    animated : if True, render stroke-by-stroke (Jupyter only)
    save_path : if given, save the final figure to this file
    """
    drawing = drawing_json["drawing"]
    word = drawing_json.get("word", "Unknown")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 255)
    ax.set_ylim(255, 0)  # invert Y
    ax.set_aspect("equal")
    ax.set_title(f"Drawing: {word}")

    for stroke in drawing:
        x_coords, y_coords = stroke
        ax.plot(x_coords, y_coords, "k-", linewidth=2)

        if animated:
            try:
                from IPython.display import clear_output, display

                clear_output(wait=True)
                display(fig)
                time.sleep(delay)
            except ImportError:
                pass  # not in Jupyter — just draw everything

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float],
    save_path: Optional[str] = None,
) -> None:
    """Plot training & validation loss + accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()

    ax2.plot(epochs, val_accs, label="Val Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
