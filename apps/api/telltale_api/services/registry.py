"""Loads trained models from TELLTALE_MODELS_DIR. Missing models are reported, never fatal."""

import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException
from telltale_ml.acoustic import audio_ops as acoustic_audio_ops
from telltale_ml.acoustic import fault as acoustic_fault
from telltale_ml.acoustic import features as acoustic_features
from telltale_ml.acoustic.detector import AcousticDetector
from telltale_ml.acoustic.fault import FaultClassifier
from telltale_ml.sensor.predictor import SensorPredictor

log = logging.getLogger("telltale.registry")


def _manifest_hash(files: list[Path]) -> str:
    """Canonical manifest hash of an explicit file set (same format as fingerprint_dir:
    sorted ``<filename> <sha256-hex>`` lines joined with ``\\n``, trailing newline)."""
    lines = []
    for file in sorted(files, key=lambda p: p.name):
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        lines.append(f"{file.name} {digest}")
    manifest = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(manifest).hexdigest()


def fingerprint_dir(path: Path) -> str:
    """Deterministic artifact fingerprint, computed ONCE per successful load.

    Canonical manifest: one line per directly-contained file, sorted by filename,
    formatted ``<filename> <sha256-hex>`` with ``\\n`` separators (UTF-8, trailing
    newline included). The fingerprint is the SHA-256 hex of that manifest.
    Callers never hash artifacts per request.
    """
    return _manifest_hash([p for p in path.iterdir() if p.is_file()])

# Shown to users in API errors and on the Models page. Commands refer to the `telltale` helper script
# in the project root (Windows: .\telltale <command>, Mac/Linux: bash telltale.sh <command>).
HINTS = {
    "sensor": "It is trained automatically on first start (about a minute). To retrain manually run: telltale train-sensor",
    "acoustic": "Train one with: telltale train-sound fan   (downloads ~1.1 GB the first time)",
    "bearing": "Train it with: telltale train-bearing   (downloads 75 MB, takes a few minutes)",
}


def _preprocessing_fingerprint(detector: AcousticDetector) -> str | None:
    """Identity of the preprocessing rules actually in force for this detector.

    Covers the effective configuration (feature config, processing cap, ensemble
    windowing), the data-dependent NORMALIZATION constants loaded from metadata
    (classical mean/std, reference mean/std), a real CODE identity (sha256 of the
    acoustic features source file that implements decode/resample/channel-select/
    log-mel) and the decoder/feature library versions in the serving environment.

    A configuration-only hash is deliberately NOT claimed as preprocessing
    provenance: if the code identity cannot be established, this returns None so
    the assessment rules flag the identity as incomplete instead of overclaiming.
    """
    try:
        feature_file = Path(inspect.getsourcefile(acoustic_features) or "")
        audio_ops_file = Path(inspect.getsourcefile(acoustic_audio_ops) or "")
        code_sha = (
            _manifest_hash([feature_file, audio_ops_file])
            if feature_file.is_file() and audio_ops_file.is_file()
            else None
        )
    except (OSError, TypeError, ValueError):
        code_sha = None
    if code_sha is None:
        return None
    import onnxruntime
    import soundfile

    meta = detector.metadata
    classical = meta.get("classical") or {}
    reference = meta.get("reference_spectrum") or {}

    def _digest(container: dict[str, Any], key: str) -> str:
        array = np.ascontiguousarray(np.asarray(container.get(key, []), dtype=np.float32))
        return hashlib.sha256(array.tobytes()).hexdigest()

    payload = {
        "features": detector.cfg.to_dict(),
        "max_audio_seconds": meta.get("max_audio_seconds"),
        "window_frames": meta.get("window_frames"),
        "window_hop_frames": meta.get("window_hop_frames"),
        "scorer": detector.kind,
        "normalization": {
            "classical_mean": _digest(classical, "mean"),
            "classical_std": _digest(classical, "std"),
            "reference_mean": _digest(reference, "mean"),
            "reference_std": _digest(reference, "std"),
        },
        "code_sha256": code_sha,
        "versions": {
            "audio_ops": acoustic_audio_ops.__version__,
            "numpy": np.__version__,
            "onnxruntime": onnxruntime.__version__,
            "soundfile": soundfile.__version__,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _bearing_preprocessing_fingerprint(classifier: FaultClassifier) -> str | None:
    """Identity of the preprocessing rules actually in force for the bearing classifier.

    Covers the effective feature configuration (sample rate, segmentation, spectral
    envelope and shaft-order features, channel select), the trained feature order
    and the model's processing cap (``max_audio_seconds``), a real CODE identity
    (sha256 of the fault-feature source implementing segmentation/feature extraction
    and of the shared audio source implementing decode/resample/channel-select), and
    the decoder/feature/library versions in the serving environment.

    A configuration-only hash is deliberately NOT claimed as preprocessing
    provenance: if the code identity cannot be established, this returns None so
    the assessment rules flag the identity as incomplete instead of overclaiming.
    """
    try:
        fault_file = Path(inspect.getsourcefile(acoustic_fault) or "")
        decode_file = Path(inspect.getsourcefile(acoustic_features) or "")
        audio_ops_file = Path(inspect.getsourcefile(acoustic_audio_ops) or "")
        fault_sha = hashlib.sha256(fault_file.read_bytes()).hexdigest() if fault_file.is_file() else None
        decode_sha = hashlib.sha256(decode_file.read_bytes()).hexdigest() if decode_file.is_file() else None
        audio_ops_sha = hashlib.sha256(audio_ops_file.read_bytes()).hexdigest() if audio_ops_file.is_file() else None
    except (OSError, TypeError, ValueError):
        fault_sha = decode_sha = audio_ops_sha = None
    if fault_sha is None or decode_sha is None or audio_ops_sha is None:
        return None
    import lightgbm
    import scipy
    import soundfile

    meta = classifier.metadata
    payload = {
        "features": classifier.cfg.to_dict(),
        "feature_names": meta.get("feature_names"),
        "max_audio_seconds": meta.get("max_audio_seconds"),
        "code_sha256": {"fault_features": fault_sha, "decode": decode_sha, "audio_ops": audio_ops_sha},
        "versions": {
            "audio_ops": acoustic_audio_ops.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": soundfile.__version__,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class ModelRegistry:
    def __init__(self, models_dir: Path, fallback_dir: Path | None = None):
        self.models_dir = Path(models_dir)
        # Where the auto-trained sensor model goes if models_dir is read-only (e.g. Linux bind-mount permissions).
        self.fallback_dir = Path(fallback_dir) if fallback_dir else None
        self.sensor: SensorPredictor | None = None
        # Identity of the SUCCESSFULLY loaded sensor artifacts (set together with
        # self.sensor in load(); None whenever no sensor model is loaded).
        self.sensor_identity: dict[str, Any] | None = None
        self.acoustic: dict[str, AcousticDetector] = {}
        # Per machine type: identity of the SUCCESSFULLY loaded acoustic artifacts
        # (artifact + preprocessing + reference fingerprints), set together with
        # self.acoustic in load(). Missing entry => identity unknown, never invented.
        self.acoustic_identity: dict[str, dict[str, Any]] = {}
        self.bearing: FaultClassifier | None = None
        # Identity of the SUCCESSFULLY loaded bearing artifacts (artifact +
        # preprocessing + reference/calibration fingerprints), set together with
        # self.bearing in load(). None whenever no bearing model is loaded.
        self.bearing_identity: dict[str, Any] | None = None
        self.errors: dict[str, str] = {}

    def load(self) -> "ModelRegistry":
        self.errors.clear()
        self.sensor_identity = None
        self.sensor = self._try("sensor", self.sensor_dir(), SensorPredictor)
        if self.sensor is not None:
            try:
                self.sensor_identity = self._sensor_identity(self.sensor_dir(), self.sensor.metadata)
            except Exception:  # fingerprinting must never break model serving
                log.exception("Could not fingerprint sensor artifacts in %s", self.sensor_dir())
                self.sensor_identity = None
        self.bearing = self._try("bearing", self.models_dir / "bearing", FaultClassifier)
        self.bearing_identity = None
        if self.bearing is not None:
            try:
                self.bearing_identity = self._bearing_identity(self.models_dir / "bearing", self.bearing)
            except Exception:  # fingerprinting must never break model serving
                log.exception("Could not fingerprint bearing artifacts in %s", self.models_dir / "bearing")
                self.bearing_identity = None
        self.acoustic = {}
        self.acoustic_identity = {}
        for meta in sorted((self.models_dir / "acoustic").glob("*/metadata.json")):
            det = self._try(f"acoustic/{meta.parent.name}", meta.parent, AcousticDetector)
            if det is not None:
                self.acoustic[det.machine_type] = det
                try:
                    self.acoustic_identity[det.machine_type] = self._acoustic_identity(meta.parent, det)
                except Exception:  # fingerprinting must never break model serving
                    log.exception("Could not fingerprint acoustic artifacts in %s", meta.parent)
        log.info(
            "Models: sensor=%s acoustic=%s bearing=%s",
            self.sensor is not None, sorted(self.acoustic), self.bearing is not None,
        )
        return self

    def sensor_dir(self) -> Path:
        primary = self.models_dir / "sensor"
        if (primary / "metadata.json").is_file() or self.fallback_dir is None:
            return primary
        return self.fallback_dir / "sensor"

    @staticmethod
    def _sensor_identity(directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        """Identity of the loaded sensor artifacts for assessment snapshots.

        Missing preprocessing identity stays explicit None: the current training
        metadata records no preprocessing version, and it is never invented.
        """
        return {
            "name": metadata.get("name") or "sensor-lightgbm",
            "version": metadata.get("version"),
            "trained_at": metadata.get("created_at"),
            "artifact_fingerprint": fingerprint_dir(directory),
            "preprocessing_fingerprint": None,
            "training_dataset": metadata.get("dataset") or metadata.get("data_source"),
        }

    @staticmethod
    def _acoustic_identity(directory: Path, detector: AcousticDetector) -> dict[str, Any]:
        """Identity of the loaded acoustic artifacts for assessment snapshots.

        - ``artifact_fingerprint``: manifest of every directly-contained artifact
          file (ONNX embedder(s), normal-condition bank, metadata). Files in the
          ``enrolled/`` subdirectory are deliberately NOT part of it - enrolled
          references are mutable runtime state and are never silently identified
          by a bare name (assessment creation is refused while enrollment is active).
        - ``preprocessing_fingerprint``: see :func:`_preprocessing_fingerprint`
          (effective rules + code + versions + normalization digests; None when
          the code identity cannot be established, never a config-only overclaim).
        - ``reference_fingerprint``: identifies the calibration/reference artifacts
          the scorer actually compares against - the normal-condition bank file
          plus the metadata (which holds the conformal calibration scores and the
          branch statistics). It is separate from the calibration LABEL ("global" /
          "enrolled:<name>") reported inside the analysis output.
        """
        meta = detector.metadata
        if detector.kind == "ensemble":
            reference_files = [directory / meta["bank_file"], directory / "metadata.json"]
        else:
            reference_files = [directory / meta["onnx"]["file"], directory / "metadata.json"]
        existing = [p for p in reference_files if p.is_file()]
        return {
            "name": meta.get("name") or f"acoustic-{meta.get('machine_type', 'unknown')}-{detector.kind}",
            "version": meta.get("version"),
            "trained_at": meta.get("created_at"),
            "artifact_fingerprint": fingerprint_dir(directory),
            "preprocessing_fingerprint": _preprocessing_fingerprint(detector),
            "training_dataset": meta.get("dataset") or meta.get("data_source"),
            "reference_fingerprint": _manifest_hash(existing) if existing else None,
        }

    @staticmethod
    def _bearing_identity(directory: Path, classifier: FaultClassifier) -> dict[str, Any]:
        """Identity of the loaded bearing artifacts for assessment snapshots.

        - ``artifact_fingerprint``: manifest of every directly-contained artifact
          file (LightGBM booster, metadata).
        - ``preprocessing_fingerprint``: see :func:`_bearing_preprocessing_fingerprint`
          (effective rules + code + versions; None when the code identity cannot be
          established, never a config-only overclaim).
        - ``reference_fingerprint``: the CONFORMAL CALIBRATION identity. The bearing
          classifier holds no separate reference bank - the calibration quantile,
          alpha, coverage and calibration-set size live inside ``metadata.json`` -
          so the immutable calibration record is that file's manifest. It is
          separate from the coverage LABEL reported inside the diagnosis output.
        """
        meta = classifier.metadata
        calibration_record = directory / "metadata.json"
        existing = [p for p in [calibration_record] if p.is_file()]
        return {
            "name": meta.get("name") or "bearing-lightgbm",
            "version": meta.get("version"),
            "trained_at": meta.get("created_at"),
            "artifact_fingerprint": fingerprint_dir(directory),
            "preprocessing_fingerprint": _bearing_preprocessing_fingerprint(classifier),
            "training_dataset": meta.get("dataset") or meta.get("data_source"),
            "reference_fingerprint": _manifest_hash(existing) if existing else None,
        }

    def _try(self, name: str, path: Path, cls):
        if not (path / "metadata.json").is_file():
            return None
        try:
            return cls(path)
        except Exception as exc:  # corrupt/incompatible artifacts must not take the API down
            log.exception("Failed to load %s model from %s", name, path)
            self.errors[name] = f"{type(exc).__name__}: {exc}"
            return None

    # ---------------------------------------------------------------- guards for routes
    def require_sensor(self) -> SensorPredictor:
        if self.sensor is None:
            raise HTTPException(503, f"Sensor model not available. {HINTS['sensor']}")
        return self.sensor

    def require_acoustic(self, machine_type: str) -> AcousticDetector:
        if machine_type not in self.acoustic:
            available = sorted(self.acoustic) or "none"
            raise HTTPException(
                503 if not self.acoustic else 404,
                f"No sound model for machine type '{machine_type}' (available: {available}). {HINTS['acoustic']}",
            )
        return self.acoustic[machine_type]

    def require_bearing(self) -> FaultClassifier:
        if self.bearing is None:
            raise HTTPException(503, f"Bearing fault model not available. {HINTS['bearing']}")
        return self.bearing

    # ---------------------------------------------------------------- introspection
    def online(self) -> dict[str, bool]:
        return {"sensor": self.sensor is not None, "acoustic": bool(self.acoustic), "bearing": self.bearing is not None}

    def status(self, detailed: bool = False) -> dict[str, Any]:
        def describe(meta: dict[str, Any]) -> dict[str, Any]:
            keep = {"created_at", "version", "data_source", "dataset", "machine_type", "threshold", "classes", "split_strategy", "scorer", "conformal"}
            out = {k: v for k, v in meta.items() if k in keep}
            if detailed:
                out["caveats"] = meta.get("caveats", [])
                if meta["kind"] == "sensor":
                    out["test_metrics"] = {"failure": meta["models"]["failure"]["test_metrics"]} | {
                        code: m["test_metrics"] for code, m in meta["models"]["modes"].items()
                    }
                    out["mode_reliability"] = {
                        code: {"val_f1": m.get("val_f1"), "reliable": m.get("reliable", True)}
                        for code, m in meta["models"]["modes"].items()
                    }
                    out["risk_thresholds"] = meta["risk_thresholds"]
                    out["feature_importance_gain"] = meta.get("feature_importance_gain", {})
                    out["n_rows"] = meta.get("n_rows")
                else:
                    out["test_metrics"] = meta.get("test_metrics", {})
            return out

        return {
            "models_dir": self.models_dir.as_posix(),
            "sensor": {"loaded": True, **describe(self.sensor.metadata)} if self.sensor else {"loaded": False, "hint": HINTS["sensor"]},
            "acoustic": {mt: {"loaded": True, **describe(d.metadata)} for mt, d in sorted(self.acoustic.items())},
            "acoustic_hint": None if self.acoustic else HINTS["acoustic"],
            "bearing": {"loaded": True, **describe(self.bearing.metadata)} if self.bearing else {"loaded": False, "hint": HINTS["bearing"]},
            "errors": self.errors,
        }
