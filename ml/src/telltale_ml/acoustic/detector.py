"""Inference for acoustic anomaly detectors (ONNX Runtime, torch-free). Used by the API.

Supports two model kinds (metadata["scorer"]):
  * "ensemble"     CNN embeddings + classical spectra, nearest-neighbour scoring, conformal p-values (default)
  * "autoencoder"  legacy reconstruction-error model (model.onnx)

Enrollment ("teach it this machine's normal sound", ensemble only): normal clips of a new condition are added
as an extra bank - no retraining. With >= MIN_CONDITION_CALIBRATION clips, that condition also gets its own
(Mondrian) conformal calibration from leave-one-out scores, which cuts false alarms under new operating conditions.
Stored in <model_dir>/enrolled/<name>.npz.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from ..utils import read_json
from .features import AudioSource, LogMelConfig, load_audio, log_mel, pool, stack_frames
from .knn_scorer import EnsembleScorer, conformal_p_value, l2_normalize, leave_one_out_stats, severity_from_p

MIN_SECONDS = 0.5
# With n calibration scores the smallest p-value is 1/(n+1); 19 makes p <= 0.05 reachable.
MIN_CONDITION_CALIBRATION = 19
ENROLL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
ENROLL_PREFIX = "user-"


def _providers() -> list[str]:
    """TELLTALE_ORT_PROVIDERS=ROCMExecutionProvider,CPUExecutionProvider to use an AMD GPU build of ORT."""
    wanted = [p.strip() for p in os.environ.get("TELLTALE_ORT_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip()]
    available = set(ort.get_available_providers())
    return [p for p in wanted if p in available] or ["CPUExecutionProvider"]


class AcousticDetector:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.metadata: dict[str, Any] = read_json(self.model_dir / "metadata.json")
        self.kind: str = self.metadata.get("scorer", "autoencoder")
        self.cfg = LogMelConfig.from_dict(self.metadata["features"])
        self.threshold = float(self.metadata["threshold"])
        self.watch_fraction = float(self.metadata.get("watch_fraction", 0.85))
        self.calibration = self.metadata.get("calibration_scores")
        self.alpha = float(self.metadata.get("alpha", 1 - float(self.metadata.get("threshold_percentile", 95)) / 100))
        self.watch_alpha = float(self.metadata.get("watch_alpha", 0.15))
        if self.kind == "ensemble":
            self.embedders = [
                (e["branch"], e["input"], ort.InferenceSession(str(self.model_dir / e["file"]), providers=_providers()))
                for e in self.metadata["embedders"]
            ]
            cl = self.metadata["classical"]
            self.cls_mean = np.asarray(cl["mean"], dtype=np.float32)
            self.cls_std = np.asarray(cl["std"], dtype=np.float32)
            self.base_scorer = EnsembleScorer.load(
                self.model_dir / self.metadata["bank_file"], self.metadata["branch_stats"], self.metadata["weights"]
            )
            self.scorer = self.base_scorer
            self._lock = threading.Lock()
            self.enrolled: dict[str, dict[str, Any]] = {}
            self._load_enrolled()
        else:
            onnx_meta = self.metadata["onnx"]
            self.session = ort.InferenceSession(str(self.model_dir / onnx_meta["file"]), providers=_providers())
            self.input_name = onnx_meta["input"]
        ref = self.metadata["reference_spectrum"]
        self.ref_band_hz = np.asarray(ref["band_hz"], dtype=np.float32)
        self.ref_mean = np.asarray(ref["mean"], dtype=np.float32)
        self.ref_std = np.maximum(np.asarray(ref["std"], dtype=np.float32), 1e-3)

    @property
    def machine_type(self) -> str:
        return self.metadata["machine_type"]

    # ------------------------------------------------------------------ status
    def status_for(self, score: float, p_value: float | None = None) -> str:
        if p_value is not None:
            if p_value <= self.alpha:
                return "anomalous"
            return "watch" if p_value <= self.watch_alpha else "normal"
        if score >= self.threshold:
            return "anomalous"
        if score >= self.threshold * self.watch_fraction:
            return "watch"
        return "normal"

    # ------------------------------------------------------------------ scoring
    def _views(self, specs: np.ndarray) -> dict[str, np.ndarray]:
        """specs: (N, n_mels, T) raw log-mel -> {'arc0': emb, ..., 'cls': standardised mean spectrum}."""
        x = specs.astype(np.float32)
        views = {branch: l2_normalize(sess.run(None, {name: x})[0]) for branch, name, sess in self.embedders}
        views["cls"] = (x.mean(axis=2) - self.cls_mean) / self.cls_std
        return views

    def _ensemble_scores(self, specs: np.ndarray) -> np.ndarray:
        """specs: (N, n_mels, T) raw log-mel -> anomaly scores (N,)."""
        return self.scorer.score(self._views(specs))

    # ------------------------------------------------------------------ enrollment
    def _enrolled_path(self, name: str) -> Path:
        return self.model_dir / "enrolled" / f"{name}.npz"

    def _load_enrolled(self) -> None:
        folder = self.model_dir / "enrolled"
        if folder.is_dir():
            for path in sorted(folder.glob("*.npz")):
                with np.load(path) as z:
                    self.enrolled[path.stem] = {
                        "views": {k[len("view__"):]: z[k] for k in z.files if k.startswith("view__")},
                        "calibration": z["calibration"] if "calibration" in z.files else np.array([]),
                    }
        self._rebuild_scorer()

    def _extra_banks(self, enrolled: dict[str, dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[float]]]]:
        banks: dict[str, np.ndarray] = {}
        stats: dict[str, dict[str, list[float]]] = {}
        for name, entry in enrolled.items():
            group = f"{ENROLL_PREFIX}{name}"
            for branch, arr in entry["views"].items():
                banks[f"{branch}__{group}__enrolled"] = arr
                if len(arr) >= 3:
                    mu, sd = leave_one_out_stats(branch, arr)
                else:  # too few clips for their own statistics: borrow the average of the trained sections
                    known = list(self.base_scorer.stats[branch].values())
                    mu, sd = float(np.mean([k[0] for k in known])), float(np.mean([k[1] for k in known]))
                stats.setdefault(branch, {})[group] = [mu, sd]
        return banks, stats

    def _rebuild_scorer(self) -> None:
        banks, stats = self._extra_banks(self.enrolled)
        scorer = self.base_scorer.with_extra(banks, stats) if banks else self.base_scorer
        with self._lock:
            self.scorer = scorer

    def enroll(self, name: str, sources: list[AudioSource]) -> dict[str, Any]:
        """Add normal clips of a new condition (e.g. a specific machine). Returns the enrollment status."""
        if self.kind != "ensemble":
            raise ValueError("Enrollment needs a v2 (ensemble) sound model - retrain with method: ensemble")
        if not ENROLL_NAME_RE.match(name):
            raise ValueError("Name must be 1-40 letters, digits, '-' or '_'")
        new_views: dict[str, list[np.ndarray]] = {}
        for src in sources:
            y = load_audio(src, self.cfg.sample_rate, self.cfg.channel, self.metadata.get("max_audio_seconds"))
            if len(y) / self.cfg.sample_rate < 1.0:
                raise ValueError("Every enrollment clip must be at least 1 second long")
            for branch, arr in self._views(log_mel(y, self.cfg)[None]).items():
                new_views.setdefault(branch, []).append(arr[0])
        old = self.enrolled.get(name, {"views": {}})["views"]
        views = {b: np.concatenate([old[b], np.stack(v)]) if b in old else np.stack(v) for b, v in new_views.items()}
        n = len(views["cls"])

        # Leave-one-out calibration: score each enrolled clip with itself removed from its own bank.
        calibration = []
        if n >= 2:
            others = {k: v for k, v in self.enrolled.items() if k != name}
            for i in range(n):
                loo = {**others, name: {"views": {b: np.delete(a, i, axis=0) for b, a in views.items()}}}
                banks, stats = self._extra_banks(loo)
                scorer = self.base_scorer.with_extra(banks, stats)
                calibration.append(float(scorer.score({b: a[i : i + 1] for b, a in views.items()})[0]))
        entry = {"views": views, "calibration": np.sort(np.asarray(calibration))}
        path = self._enrolled_path(name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, calibration=entry["calibration"], **{f"view__{b}": a for b, a in views.items()})
        except OSError as exc:
            raise ValueError(f"Could not save the enrolled clips in {path.parent} ({exc.strerror}). "
                             "Make sure the models folder is writable.") from exc
        self.enrolled[name] = entry
        self._rebuild_scorer()
        return self.enrollment_status(name)

    def enrollment_status(self, name: str) -> dict[str, Any]:
        entry = self.enrolled.get(name)
        n = 0 if entry is None else len(entry["views"]["cls"])
        n_cal = 0 if entry is None else len(entry["calibration"])
        return {
            "name": name,
            "clips": n,
            "calibrated": n_cal >= MIN_CONDITION_CALIBRATION,
            "clips_needed_for_calibration": max(0, MIN_CONDITION_CALIBRATION - n_cal),
        }

    def clear_enrollment(self, name: str) -> bool:
        if name not in self.enrolled:
            return False
        self._enrolled_path(name).unlink(missing_ok=True)
        del self.enrolled[name]
        self._rebuild_scorer()
        return True

    def _p_value(self, score: float, views: dict[str, np.ndarray] | None) -> tuple[float | None, str]:
        """Mondrian conformal: use the nearest condition's own calibration when it has enough scores."""
        if not self.calibration:
            return None, "none"
        if self.kind == "ensemble" and views is not None and self.enrolled:
            group = self.scorer.nearest_condition(views)[0].split("__")[0]
            if group.startswith(ENROLL_PREFIX):
                name = group[len(ENROLL_PREFIX):]
                cal = self.enrolled.get(name, {}).get("calibration", [])
                if len(cal) >= MIN_CONDITION_CALIBRATION:
                    return conformal_p_value(score, cal), f"enrolled:{name}"
        return conformal_p_value(score, self.calibration), "global"

    def _score_and_timeline(self, lm: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        hop_s = self.cfg.hop_length / self.cfg.sample_rate
        if self.kind == "ensemble":
            score = float(self.scorer.score(self._views(lm[None]))[0])
            win = int(self.metadata.get("window_frames", 64))
            hop = int(self.metadata.get("window_hop_frames", 16))
            if lm.shape[1] <= win:
                return score, np.array([score]), np.array([lm.shape[1] * hop_s / 2])
            starts = range(0, lm.shape[1] - win + 1, hop)
            windows = np.stack([lm[:, s : s + win] for s in starts])
            values = np.concatenate([self._ensemble_scores(windows[i : i + 64]) for i in range(0, len(windows), 64)])
            times = np.array([(s + win / 2) * hop_s for s in starts])
            return score, values, times
        errors = self.session.run(None, {self.input_name: stack_frames(lm, self.cfg.frames)})[0]
        return float(errors.mean()), errors, np.arange(len(errors)) * hop_s

    def analyze(self, source: AudioSource, max_points: int = 200, max_bands: int = 64) -> dict[str, Any]:
        cfg = self.cfg
        y = load_audio(source, cfg.sample_rate, cfg.channel, self.metadata.get("max_audio_seconds"))
        duration = len(y) / cfg.sample_rate
        if duration < MIN_SECONDS:
            raise ValueError(f"Audio too short ({duration:.2f}s); need at least {MIN_SECONDS}s")

        lm = log_mel(y, cfg)
        score, frame_values, frame_times = self._score_and_timeline(lm)
        p_value, calibration = self._p_value(score, self._views(lm[None]) if self.kind == "ensemble" else None)
        status = self.status_for(score, p_value)
        severity = severity_from_p(p_value, self.alpha) if p_value is not None else (score / self.threshold if self.threshold > 0 else 0.0)

        frame_err = pool(frame_values, max_points, axis=0)
        frame_t = pool(frame_times, max_points, axis=0)
        hop_s = cfg.hop_length / cfg.sample_rate
        spec = pool(pool(lm, max_bands, axis=0), max_points, axis=1)
        spec_t = pool(np.arange(lm.shape[1]) * hop_s, max_points, axis=0)
        band_hz = pool(self.ref_band_hz, max_bands, axis=0)

        z = (lm.mean(axis=1) - self.ref_mean) / self.ref_std
        dev_hz = pool(self.ref_band_hz, 32, axis=0)
        dev_z = pool(z, 32, axis=0)

        n_buckets = min(600, len(y))
        buckets = np.array_split(y, n_buckets)
        wf_min = np.array([b.min() for b in buckets])
        wf_max = np.array([b.max() for b in buckets])

        peak_idx = int(np.argmax(frame_err))
        return {
            "machine_type": self.machine_type,
            "method": self.kind,
            "score": score,
            "threshold": self.threshold,
            "severity": float(severity),
            "p_value": p_value,
            # "more unusual than X% of normal recordings"
            "unusual_percentile": None if p_value is None else float(100 * (1 - p_value)),
            "calibration": calibration,  # "global" | "enrolled:<name>" | "none"
            "status": status,
            "is_anomalous": status == "anomalous",
            "duration_s": duration,
            "sample_rate": cfg.sample_rate,
            "peak_time_s": float(frame_t[peak_idx]),
            "frame_errors": {"times": np.round(frame_t, 3).tolist(), "values": np.round(frame_err, 4).tolist()},
            "spectrogram": {
                "times": np.round(spec_t, 3).tolist(),
                "band_hz": np.round(band_hz, 1).tolist(),
                "values": np.round(spec, 2).tolist(),  # [band][time], dB
            },
            "band_deviation": {"band_hz": np.round(dev_hz, 1).tolist(), "z": np.round(dev_z, 3).tolist()},
            "waveform": {
                "duration_s": duration,
                "min": np.round(wf_min, 4).tolist(),
                "max": np.round(wf_max, 4).tolist(),
            },
        }
