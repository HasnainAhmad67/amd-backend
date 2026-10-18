"""Supervised bearing-fault classification from sound: features + inference (torch-free)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis

from ..utils import read_json
from . import audio_ops
from .features import AudioSource, load_audio

# Every feature is loudness-independent: recordings from different microphones/setups differ mostly in
# level, and a model that keys on volume learns the recording setup instead of the bearing condition.
SCALAR_FEATURES = (
    "crest_factor",  # peak / rms: impulsive faults (spalls, cracks) raise it
    "kurtosis",  # classic bearing-defect indicator
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
)


@dataclass(frozen=True)
class FaultFeatureConfig:
    sample_rate: int = 44_100
    segment_seconds: float = 1.0
    hop_seconds: float = 0.5
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 64
    channel: int = 0
    # Envelope spectrum: band-pass around the structural resonance, demodulate (Hilbert), and look at
    # which repetition rates (5-1000 Hz) dominate - that is where bearing-defect "ticks" show up.
    envelope_band_hz: tuple[float, float] = (1000.0, 10000.0)
    envelope_bins: int = 32
    # Order spectrum: the same envelope spectrum on a frequency axis divided by the shaft speed (rpm / 60).
    # Defect tick rates are fixed multiples ("orders") of the shaft speed, so they line up across speeds.
    # Needs the rpm; 0 bins = off. When the rpm is unknown, `default_rpm` is assumed (reported to the user).
    order_bins: int = 0
    order_range: tuple[float, float] = (0.5, 30.0)
    default_rpm: float = 1200.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FaultFeatureConfig:
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        for key in ("envelope_band_hz", "order_range"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def feature_names(cfg: FaultFeatureConfig) -> list[str]:
    return (
        [f"mel_shape_{i:02d}" for i in range(cfg.n_mels)]  # band level minus the segment's average level
        + [f"mel_std_{i:02d}" for i in range(cfg.n_mels)]
        + list(SCALAR_FEATURES)
        + [f"envelope_{i:02d}" for i in range(cfg.envelope_bins)]
        + [f"order_{i:02d}" for i in range(cfg.order_bins)]
    )


def _binned_log_shape(spec: np.ndarray, axis: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean power per bin -> normalised log shape (independent of loudness). Empty bins get a tiny value."""
    bins = np.array([
        spec[m].mean() if np.any(m := (axis >= a) & (axis < b)) else 1e-20
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ])
    bins = np.log(bins / (bins.sum() + 1e-20) + 1e-12)
    return bins - bins.mean()


def _envelope_spectra(
    seg: np.ndarray, cfg: FaultFeatureConfig, sos: np.ndarray, edges_hz: np.ndarray, edges_orders: np.ndarray | None,
    rpm: float,
) -> list[np.ndarray]:
    """Envelope spectrum binned in Hz and (optionally) in shaft orders."""
    env = np.abs(hilbert(sosfiltfilt(sos, seg)))
    env -= env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    freqs = np.fft.rfftfreq(len(env), 1 / cfg.sample_rate)
    out = [_binned_log_shape(spec, freqs, edges_hz)]
    if edges_orders is not None:
        out.append(_binned_log_shape(spec, freqs / (rpm / 60.0), edges_orders))
    return out


def segment_features(y: np.ndarray, cfg: FaultFeatureConfig, rpm: float | None = None) -> np.ndarray:
    """Waveform -> (n_segments, n_features). Segments are ``segment_seconds`` long, ``hop_seconds`` apart.

    ``rpm`` (shaft speed) is only used for the order spectrum; ``cfg.default_rpm`` if not given.
    """
    rpm = float(rpm or cfg.default_rpm)
    seg_len = int(cfg.segment_seconds * cfg.sample_rate)
    hop = int(cfg.hop_seconds * cfg.sample_rate)
    if len(y) < seg_len:
        raise ValueError(f"Audio too short: need >= {cfg.segment_seconds}s")
    lo, hi = cfg.envelope_band_hz
    sos = butter(4, [lo, min(hi, 0.45 * cfg.sample_rate)], btype="bandpass", fs=cfg.sample_rate, output="sos")
    edges = np.geomspace(5, 1000, cfg.envelope_bins + 1)
    edges_orders = np.geomspace(*cfg.order_range, cfg.order_bins + 1) if cfg.order_bins else None
    rows = []
    for start in range(0, len(y) - seg_len + 1, hop):
        seg = y[start : start + seg_len]
        power = np.abs(audio_ops.stft(seg, n_fft=cfg.n_fft, hop_length=cfg.hop_length)) ** 2
        mag = np.sqrt(power)
        mel = audio_ops.power_to_db(
            audio_ops.melspectrogram(S=power, sr=cfg.sample_rate, n_mels=cfg.n_mels)
        )
        rms = float(np.sqrt(np.mean(seg**2)) + 1e-12)
        band_level = mel.mean(axis=1)
        rows.append(
            np.concatenate(
                [
                    band_level - band_level.mean(),  # spectral shape (dB offsets cancel out)
                    mel.std(axis=1),  # variation over time per band (also level-independent in dB)
                    [
                        float(np.max(np.abs(seg)) / rms),
                        float(kurtosis(seg)),
                        float(audio_ops.zero_crossing_rate(seg).mean()),
                        float(audio_ops.spectral_centroid(S=mag, sr=cfg.sample_rate).mean()),
                        float(audio_ops.spectral_bandwidth(S=mag, sr=cfg.sample_rate).mean()),
                        float(audio_ops.spectral_rolloff(S=mag, sr=cfg.sample_rate).mean()),
                        float(audio_ops.spectral_flatness(S=mag).mean()),
                    ],
                    *_envelope_spectra(seg, cfg, sos, edges, edges_orders, rpm),
                ]
            )
        )
    return np.asarray(rows, dtype=np.float32)


def label_from_path(rel_path: Path, keywords: dict[str, list[str]], priority: tuple[str, ...] = ()) -> str | None:
    """Match exactly one class from folder/file names. Keywords <= 3 chars must match a whole token.

    Classes in ``priority`` win when they match together with others, e.g. "Combined faults (Inner, outer, ball)"
    is the "combined" class, not ambiguous.
    """
    text = rel_path.as_posix().lower()
    tokens = set(re.split(r"[^a-z0-9]+", text))
    hits = {
        label
        for label, kws in keywords.items()
        if any((kw.lower() in tokens) if len(kw) <= 3 else (kw.lower() in text) for kw in kws)
    }
    for label in priority:
        if label in hits:
            return label
    return hits.pop() if len(hits) == 1 else None


class FaultClassifier:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.metadata: dict[str, Any] = read_json(self.model_dir / "metadata.json")
        self.cfg = FaultFeatureConfig.from_dict(self.metadata["features"])
        self.classes: list[str] = self.metadata["classes"]
        self.class_labels: dict[str, str] = self.metadata.get("class_labels", {})
        self.booster = lgb.Booster(model_file=str(self.model_dir / self.metadata["model_file"]))
        conformal = self.metadata.get("conformal") or {}
        self.conformal_q: float | None = conformal.get("q")
        self.conformal_coverage: float | None = conformal.get("coverage")

    @property
    def uses_rpm(self) -> bool:
        return self.cfg.order_bins > 0

    def diagnose(self, source: AudioSource, rpm: float | None = None) -> dict[str, Any]:
        y = load_audio(source, self.cfg.sample_rate, self.cfg.channel, self.metadata.get("max_audio_seconds"))
        rpm_assumed = self.uses_rpm and not rpm
        feats = segment_features(y, self.cfg, rpm)
        proba = np.asarray(self.booster.predict(feats)).reshape(len(feats), len(self.classes))
        mean = proba.mean(axis=0)
        best = int(mean.argmax())
        agreement = float((proba.argmax(axis=1) == best).mean())
        # Conformal prediction set: every class whose score is within the calibrated quantile.
        # Contains the true class ~coverage of the time (under similar recording conditions).
        if self.conformal_q is not None:
            pred_set = [c for c, p in zip(self.classes, mean, strict=True) if 1.0 - p <= self.conformal_q]
            pred_set = sorted(pred_set, key=lambda c: -mean[self.classes.index(c)]) or [self.classes[best]]
        else:
            pred_set = [self.classes[best]]
        return {
            "prediction_set": pred_set,
            "prediction_set_labels": [self.class_labels.get(c, c) for c in pred_set],
            "coverage": self.conformal_coverage,
            "uncertain": len(pred_set) > 1,
            "rpm_used": (float(rpm) if rpm else self.cfg.default_rpm) if self.uses_rpm else None,
            "rpm_assumed": bool(rpm_assumed),
            "predicted_class": self.classes[best],
            "label": self.class_labels.get(self.classes[best], self.classes[best]),
            "confidence": float(mean[best]),
            "segment_agreement": agreement,
            "segments": int(len(feats)),
            "duration_s": len(y) / self.cfg.sample_rate,
            "probabilities": sorted(
                (
                    {"class_name": c, "label": self.class_labels.get(c, c), "probability": float(p)}
                    for c, p in zip(self.classes, mean, strict=True)
                ),
                key=lambda d: d["probability"],
                reverse=True,
            ),
        }
