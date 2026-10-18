"""Torch models (training only — the API never imports this module)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _block(i: int, o: int) -> list[nn.Module]:
    return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]


class DenseAutoencoder(nn.Module):
    """DCASE-baseline-style autoencoder: 4x hidden -> bottleneck -> 4x hidden -> reconstruction."""

    def __init__(self, input_dim: int, hidden: int = 128, bottleneck: int = 8, depth: int = 4):
        super().__init__()
        enc: list[nn.Module] = []
        dims = [input_dim] + [hidden] * depth
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            enc += _block(a, b)
        enc += _block(hidden, bottleneck)
        dec: list[nn.Module] = _block(bottleneck, hidden)
        for _ in range(depth - 1):
            dec += _block(hidden, hidden)
        dec.append(nn.Linear(hidden, input_dim))
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class ScoringModel(nn.Module):
    """Raw feature vectors -> per-vector reconstruction MSE.

    Normalisation is baked in so the exported ONNX graph is self-contained:
    input ``features`` (N, D) float32  ->  output ``vector_error`` (N,) float32.
    """

    def __init__(self, autoencoder: DenseAutoencoder, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.autoencoder = autoencoder
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.std
        return ((z - self.autoencoder(z)) ** 2).mean(dim=1)


def pick_device(requested: str = "auto") -> torch.device:
    """ROCm builds of PyTorch expose AMD GPUs through the ``torch.cuda`` API, so "cuda" == AMD GPU there."""
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
