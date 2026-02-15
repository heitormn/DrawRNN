"""
DrawRNN Model — LSTM-based classifier for Quick, Draw! sketches.
"""

import torch
import torch.nn as nn


class DrawRNNModel(nn.Module):
    """
    Multi-layer LSTM that reads a variable-length sequence of (x, y, stroke_progress)
    points and produces a classification logit vector.

    Parameters
    ----------
    input_size : int
        Number of features per time-step (default 3: x, y, stroke_progress).
    hidden_size : int
        Number of LSTM hidden units.
    output_size : int
        Number of target classes.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout applied between LSTM layers (ignored when num_layers == 1).
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 128,
        output_size: int = 3,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_size)

        Returns
        -------
        logits : Tensor of shape (batch, output_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        logits = self.fc(out[:, -1, :])  # last time-step
        return logits
