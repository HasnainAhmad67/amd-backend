"""Audio loading + log-mel features shared by training and serving. Never duplicate elsewhere."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import IO, Any

import numpy as np
import soundfile as sf

from . import audio_ops

AudioSource = str | Path | IO[bytes]

# The Mendeley bearing dataset ships MATLAB files: one variable "x" scaled to [-1, 1], recorded at 44.1 kHz
# (the files themselves carry no sample rate).
MAT_DEFAULT_SAMPLE_RATE = 44_100
MAT_MAGIC = b"MATLAB"


@dataclass(frozen=True)
class LogMelConfig:
    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    power: float = 2.0
    frames: int = 5  # consecutive frames stacked into one input vector
    channel: int = 0  # MIMII is 8-channel; MIMII DUE is mono

    @property
    def input_dim(self) -> int:
        return self.n_mels * self.frames

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LogMelConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_mat(source: AudioSource) -> bool:
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower() == ".mat"
    pos = source.tell()
    head = source.read(len(MAT_MAGIC))
    source.seek(pos)
    return head == MAT_MAGIC


def _read_mat(source: AudioSource) -> tuple[np.ndarray, int]:
    """MATLAB v5 file -> (samples x channels float32, sample_rate). Picks the largest numeric array."""
    from scipy.io import loadmat

    mat = loadmat(source)
    sr = MAT_DEFAULT_SAMPLE_RATE
    for key in ("fs", "Fs", "FS", "sr", "sample_rate", "SampleRate"):
        if key in mat and np.size(mat[key]) == 1:
            sr = int(np.squeeze(mat[key]))
            break
    arrays = [
        np.asarray(v)
        for k, v in mat.items()
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.dtype.kind in "fiu" and v.size > 1000
    ]
    if not arrays:
        raise ValueError("No audio signal found in the .mat file")
    data = max(arrays, key=lambda a: a.size).astype(np.float32)
    if data.ndim == 1:
        data = data[:, None]
    elif data.shape[0] < data.shape[1]:
        data = data.T  # channels were stored as rows
    return data, sr


def load_audio(source: AudioSource, sample_rate: int, channel: int = 0, max_seconds: float | None = None) -> np.ndarray:
    """Read wav/flac/ogg (anything libsndfile supports) or a MATLAB .mat recording -> mono float32 at ``sample_rate``."""
    if _is_mat(source):
        data, sr = _read_mat(source)
    else:
        data, sr = sf.read(source, dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError("Audio file is empty")
    y = data[:, min(channel, data.shape[1] - 1)]
    if max_seconds is not None:
        y = y[: int(max_seconds * sr)]
    if sr != sample_rate:
        y = audio_ops.resample(y, orig_sr=sr, target_sr=sample_rate)
    return np.ascontiguousarray(y, dtype=np.float32)


def log_mel(y: np.ndarray, cfg: LogMelConfig) -> np.ndarray:
    """(n_mels, T) log-mel spectrogram in dB-like units (DCASE baseline convention)."""
    mel = audio_ops.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels, power=cfg.power
    )
    return (20.0 / cfg.power * np.log10(np.maximum(mel, 1e-10))).astype(np.float32)


def stack_frames(logmel: np.ndarray, frames: int) -> np.ndarray:
    """(n_mels, T) -> (T - frames + 1, n_mels * frames). Vector t = frames t..t+frames-1 concatenated."""
    n_mels, n_t = logmel.shape
    n_vec = n_t - frames + 1
    if n_vec < 1:
        return np.empty((0, n_mels * frames), dtype=np.float32)
    out = np.empty((n_vec, n_mels * frames), dtype=np.float32)
    for t in range(frames):
        out[:, n_mels * t : n_mels * (t + 1)] = logmel[:, t : t + n_vec].T
    return out


def mel_band_hz(cfg: LogMelConfig) -> np.ndarray:
    """Centre frequency (Hz) of each mel band, for labelling plots."""
    return audio_ops.mel_frequencies(n_mels=cfg.n_mels + 2, fmin=0.0, fmax=cfg.sample_rate / 2)[1:-1]


def pool(arr: np.ndarray, target: int, axis: int) -> np.ndarray:
    """Mean-pool ``arr`` along ``axis`` down to at most ``target`` bins (for compact API payloads)."""
    n = arr.shape[axis]
    if n <= target:
        return arr
    chunks = np.array_split(np.arange(n), target)
    return np.stack([np.take(arr, idx, axis=axis).mean(axis=axis) for idx in chunks], axis=axis)
