"""Small inference-time audio primitives implemented with NumPy and SciPy.

This module replaces the broad librosa dependency in the serverless runtime. The
formulas follow librosa's defaults for centered Hann STFTs, Slaney mel filters,
mel-band centers, and common spectral descriptors. Training-only packages are
kept out of the Vercel function bundle.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly

__version__ = "1.0.0"

_MEL_F_SP = 200.0 / 3.0
_MEL_MIN_LOG_HZ = 1_000.0
_MEL_MIN_LOG = 15.0
_MEL_LOG_STEP = math.log(6.4) / 27.0


def _hz_to_mel(frequencies: np.ndarray | float) -> np.ndarray:
    hz = np.asarray(frequencies, dtype=np.float64)
    mel = hz / _MEL_F_SP
    log_region = hz >= _MEL_MIN_LOG_HZ
    mel = np.where(
        log_region,
        _MEL_MIN_LOG + np.log(np.maximum(hz, _MEL_MIN_LOG_HZ) / _MEL_MIN_LOG_HZ) / _MEL_LOG_STEP,
        mel,
    )
    return mel


def _mel_to_hz(mels: np.ndarray | float) -> np.ndarray:
    mel = np.asarray(mels, dtype=np.float64)
    hz = _MEL_F_SP * mel
    log_region = mel >= _MEL_MIN_LOG
    hz = np.where(
        log_region,
        _MEL_MIN_LOG_HZ * np.exp(_MEL_LOG_STEP * (mel - _MEL_MIN_LOG)),
        hz,
    )
    return hz


def mel_frequencies(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Return Slaney-spaced mel-band centre frequencies in Hz."""
    points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels, dtype=np.float64)
    return _mel_to_hz(points)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float = 0.0) -> np.ndarray:
    fmax = sample_rate / 2.0
    mel_edges = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2, dtype=np.float64)
    hz_edges = _mel_to_hz(mel_edges)
    fft_hz = np.linspace(0.0, fmax, 1 + n_fft // 2, dtype=np.float64)
    lower = (fft_hz[None, :] - hz_edges[:-2, None]) / (hz_edges[1:-1, None] - hz_edges[:-2, None])
    upper = (hz_edges[2:, None] - fft_hz[None, :]) / (hz_edges[2:, None] - hz_edges[1:-1, None])
    weights = np.maximum(0.0, np.minimum(lower, upper))
    # librosa's default Slaney normalization gives each filter unit area.
    weights *= (2.0 / np.maximum(hz_edges[2:] - hz_edges[:-2], np.finfo(float).tiny))[:, None]
    return weights.astype(np.float32)


def stft(y: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Centered, zero-padded, periodic-Hann STFT (frequency x frame)."""
    samples = np.asarray(y, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("STFT input must be one-dimensional")
    if samples.size == 0:
        return np.empty((1 + n_fft // 2, 0), dtype=np.complex64)
    pad = n_fft // 2
    padded = np.pad(samples, (pad, pad), mode="constant")
    if padded.size < n_fft:
        padded = np.pad(padded, (0, n_fft - padded.size), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    return np.fft.rfft(frames * window[None, :], n=n_fft, axis=1).T.astype(np.complex64)


def melspectrogram(
    y: np.ndarray | None = None,
    *,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    power: float = 2.0,
    S: np.ndarray | None = None,
) -> np.ndarray:
    """Mel spectrogram with librosa-compatible Slaney filters and centered STFT."""
    if S is None:
        if y is None:
            raise ValueError("Either waveform y or spectrogram S must be provided")
        spectrum = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length)) ** power
    else:
        spectrum = np.asarray(S, dtype=np.float32)
        n_fft = 2 * (spectrum.shape[0] - 1)
    return _mel_filterbank(sr, n_fft, n_mels) @ spectrum


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling using SciPy; preserves sample count to within one sample."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")
    if orig_sr == target_sr:
        return np.asarray(y, dtype=np.float32)
    divisor = math.gcd(int(orig_sr), int(target_sr))
    return resample_poly(np.asarray(y, dtype=np.float32), target_sr // divisor, orig_sr // divisor).astype(np.float32)


def power_to_db(power: np.ndarray, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    """Convert a power spectrogram to dB, with librosa's default 80 dB clipping."""
    values = 10.0 * np.log10(np.maximum(np.asarray(power), amin))
    if top_db is not None and values.size:
        values = np.maximum(values, np.max(values) - top_db)
    return values


def _centered_frames(y: np.ndarray, frame_length: int, hop_length: int, edge: bool = False) -> np.ndarray:
    samples = np.asarray(y, dtype=np.float64)
    pad = frame_length // 2
    padded = np.pad(samples, (pad, pad), mode="edge" if edge and samples.size else "constant")
    if padded.size < frame_length:
        padded = np.pad(padded, (0, frame_length - padded.size), mode="constant")
    return np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]


def zero_crossing_rate(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    frames = _centered_frames(y, frame_length, hop_length, edge=True)
    # Match librosa's zero_pos=True convention: zeros are treated as positive.
    signs = frames >= 0
    return (np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1) / max(1, frame_length - 1))[None, :]


def _spectral_magnitude(S: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(S, dtype=np.float64))


def spectral_centroid(S: np.ndarray, sr: int, n_fft: int | None = None) -> np.ndarray:
    magnitude = _spectral_magnitude(S)
    n_fft = n_fft or 2 * (magnitude.shape[0] - 1)
    frequencies = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2)[:, None]
    norm = magnitude.sum(axis=0, keepdims=True)
    return ((frequencies * magnitude).sum(axis=0, keepdims=True) / np.maximum(norm, 1e-20))


def spectral_bandwidth(S: np.ndarray, sr: int, n_fft: int | None = None, p: float = 2.0) -> np.ndarray:
    magnitude = _spectral_magnitude(S)
    centroid = spectral_centroid(magnitude, sr, n_fft)
    n_fft = n_fft or 2 * (magnitude.shape[0] - 1)
    frequencies = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2)[:, None]
    norm = magnitude.sum(axis=0, keepdims=True)
    moments = (magnitude * np.abs(frequencies - centroid) ** p).sum(axis=0, keepdims=True)
    return (moments / np.maximum(norm, 1e-20)) ** (1.0 / p)


def spectral_rolloff(S: np.ndarray, sr: int, n_fft: int | None = None, roll_percent: float = 0.85) -> np.ndarray:
    magnitude = _spectral_magnitude(S)
    n_fft = n_fft or 2 * (magnitude.shape[0] - 1)
    frequencies = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2)
    cumulative = np.cumsum(magnitude, axis=0)
    threshold = roll_percent * cumulative[-1:, :]
    indices = np.argmax(cumulative >= threshold, axis=0)
    return frequencies[indices][None, :]


def spectral_flatness(S: np.ndarray, power: float = 2.0, amin: float = 1e-10) -> np.ndarray:
    values = np.maximum(_spectral_magnitude(S) ** power, amin)
    return (np.exp(np.mean(np.log(values), axis=0)) / np.mean(values, axis=0))[None, :]
