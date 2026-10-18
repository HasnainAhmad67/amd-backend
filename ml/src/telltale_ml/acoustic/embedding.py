"""Discriminative embeddings for anomalous sound detection (training only - needs torch).

Idea (DCASE 2021+ state of the art): instead of reconstructing normal sounds, train a CNN to tell apart the
recording *conditions* it sees in normal data (section x domain). An anomaly then lands far from every normal
cluster in embedding space. The angular-margin (ArcFace) loss makes those clusters compact and well separated.

The exported ONNX graph takes a raw log-mel spectrogram (N, n_mels, T) - any T - and returns L2-normalised
embeddings (N, emb_dim). Per-band normalisation is baked in, so inference code needs nothing else.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..utils import get_logger

log = get_logger("telltale.acoustic.embedding")


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SpecEmbedder(nn.Module):
    """Small 2-D CNN over (freq, time) -> fixed-size embedding. Works for any clip length >= 16 frames."""

    def __init__(self, n_mels: int, emb_dim: int = 128, widths: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        chans = (1, *widths)
        self.body = nn.Sequential(*[_block(a, b) for a, b in zip(chans[:-1], chans[1:], strict=True)])
        self.proj = nn.Linear(widths[-1] * 2, emb_dim)
        self.register_buffer("band_mean", torch.zeros(1, n_mels, 1))
        self.register_buffer("band_std", torch.ones(1, n_mels, 1))

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:  # (N, n_mels, T)
        x = (logmel - self.band_mean) / self.band_std
        h = self.body(x[:, None]).mean(dim=2)  # pool frequency -> (N, C, T')
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)  # mean + max over time
        return self.proj(h)


class Normalized(nn.Module):
    """Export wrapper: L2-normalised embeddings (cosine distance = 1 - dot product)."""

    def __init__(self, net: SpecEmbedder):
        super().__init__()
        self.net = net

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(logmel), dim=1)


class ArcFace(nn.Module):
    """Additive angular margin softmax (Deng et al., 2019)."""

    def __init__(self, emb_dim: int, n_classes: int, scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, emb_dim) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, emb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(emb), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cos) + self.margin)
        onehot = F.one_hot(y, cos.shape[1]).float()
        return F.cross_entropy(self.scale * (onehot * target + (1 - onehot) * cos), y)


def train_embedder(
    specs: np.ndarray,  # (N, n_mels, T) float32 log-mel of NORMAL training clips
    labels: np.ndarray,  # (N,) int class ids (section x domain)
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
) -> SpecEmbedder:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    net = SpecEmbedder(specs.shape[1], int(cfg.get("emb_dim", 128)))
    net.band_mean.copy_(torch.from_numpy(specs.mean(axis=(0, 2), keepdims=True)[0][None]))
    net.band_std.copy_(torch.from_numpy(specs.std(axis=(0, 2), keepdims=True)[0][None] + 1e-5))
    head = ArcFace(net.proj.out_features, n_classes, float(cfg.get("arcface_scale", 30.0)), float(cfg.get("arcface_margin", 0.5)))
    net.to(device)
    head.to(device)

    epochs, batch = int(cfg.get("epochs", 20)), int(cfg.get("batch_size", 64))
    crop = min(int(cfg.get("crop_frames", 192)), specs.shape[2])
    steps = max(1, len(specs) // batch)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(cfg.get("learning_rate", 2e-3)), total_steps=epochs * steps)
    # Oversample rare classes (e.g. 3 target-domain clips vs 1000 source clips per section).
    weights = 1.0 / np.bincount(labels)[labels]
    weights /= weights.sum()
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        head.train()
        order = rng.choice(len(specs), size=steps * batch, p=weights)
        total = 0.0
        for b in range(steps):
            idx = order[b * batch : (b + 1) * batch]
            starts = rng.integers(0, specs.shape[2] - crop + 1, size=len(idx))
            xb = torch.from_numpy(np.stack([specs[i, :, s : s + crop] for i, s in zip(idx, starts, strict=True)]))
            if rng.random() < 0.5:  # SpecAugment-style frequency mask (on normalised scale -> set to band mean)
                f0 = int(rng.integers(0, specs.shape[1] - 16))
                xb[:, f0 : f0 + int(rng.integers(4, 16)), :] = net.band_mean.cpu()[0, f0 : f0 + 1, 0]
            loss = head(net(xb.to(device)), torch.from_numpy(labels[idx]).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        log.info("embedder epoch %d/%d loss %.3f (%.0fs)", ep + 1, epochs, total / steps, time.time() - t0)
    return net.eval()


def export_embedder(net: SpecEmbedder, n_mels: int, path) -> None:
    model = Normalized(net).cpu().eval()
    dummy = torch.randn(2, n_mels, 64) * 10 - 40
    kwargs: dict[str, Any] = dict(
        input_names=["logmel"],
        output_names=["embedding"],
        dynamic_axes={"logmel": {0: "n", 2: "t"}, "embedding": {0: "n"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:  # torch < 2.5
        torch.onnx.export(model, (dummy,), str(path), **kwargs)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for t in (64, 313):  # check several lengths: dynamic time axis must work
        sample = torch.randn(3, n_mels, t) * 10 - 40
        ref = model(sample).detach().numpy()
        got = sess.run(None, {"logmel": sample.numpy()})[0]
        if not np.allclose(ref, got, atol=1e-4):
            raise RuntimeError(f"ONNX embedder mismatch at T={t}: max diff {np.abs(ref - got).max()}")
