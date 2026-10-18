"""Train the supervised bearing-fault classifier on the Mendeley bearing-sound dataset.

    telltale-train-bearing                     # downloads the 75 MB dataset on first run
    telltale-train-bearing --data-root path/to/extracted/dataset

How it is evaluated (segments of one recording overlap, so recordings must never be split):
  * Each recording gets a *condition* = recording setup + speed band, parsed from its path
    (e.g. "1-1200" = Datasets 1 / stethoscope at ~1200 rpm). See `evaluation:` in configs/bearing.yaml.
  * Headline metrics: leave-one-condition-out cross-validation -> "how well does it do at a speed/setup
    combination it has never heard?"
  * Also reported: leave-one-setup-out (train with the stethoscope, test without, and vice versa) ->
    the much harder "different microphone" case.
  * If conditions can't be parsed (other datasets), it falls back to leave-one-recording-out.
  * The final model is trained on every condition except `demo_holdout`; those recordings become the
    UI's demo clips, so demo predictions are on audio the model never saw.

Robustness to new recordings:
  * Microphone augmentation: training copies with a random EQ curve, gain and noise (only ever added to the
    TRAINING side of a fold) so the model learns to ignore the microphone's colouring.
  * Conformal prediction sets (split conformal on out-of-fold predictions): the API returns every class that
    can't be ruled out at the configured coverage, so an unsure model says "inner race or ball".

Artifacts written to <models>/bearing/: model.txt, metadata.json   (+ data/samples/bearing_<class>_0.wav)
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .. import __version__
from ..paths import data_dir, default_config, models_dir, raw_dir
from ..utils import get_logger, load_config, seed_everything, utc_now, write_json
from .fault import FaultFeatureConfig, feature_names, label_from_path, segment_features
from .features import load_audio

log = get_logger("telltale.acoustic.bearing")
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mat"}


def samples_dir() -> Path:
    return data_dir() / "samples"


@dataclass
class Recording:
    path: Path
    label: str
    condition: str | None
    features: np.ndarray  # (n_segments, n_features)
    augmented: list[np.ndarray] | None = None  # features of simulated-microphone copies (training only)


def random_microphone(y: np.ndarray, sr: int, rng: np.random.Generator, aug: dict[str, Any]) -> np.ndarray:
    """Simulate another microphone/setup: smooth random EQ curve, gain change and a little noise."""
    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    knots = np.geomspace(50, sr / 2, 10)
    eq_db = float(aug.get("eq_db", 9.0))
    curve = np.interp(np.log(np.maximum(freqs, 1.0)), np.log(knots), rng.uniform(-eq_db, eq_db, size=len(knots)))
    out = np.fft.irfft(spectrum * 10 ** (curve / 20), n=len(y))
    lo, hi = aug.get("gain_db", [-12, 6])
    out *= 10 ** (rng.uniform(lo, hi) / 20)
    nlo, nhi = aug.get("noise", [0.001, 0.01])
    out += rng.normal(0, rng.uniform(nlo, nhi) * (np.std(out) + 1e-9), len(out))
    return out.astype(np.float32)


# --------------------------------------------------------------------------- discovery
def discover_labelled(root: Path, class_cfg: dict[str, Any]) -> tuple[list[tuple[Path, str]], list[str]]:
    keywords = {name: c["keywords"] for name, c in class_cfg.items()}
    priority = tuple(name for name, c in class_cfg.items() if c.get("match_first"))
    labelled, unlabelled = [], []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT):
        label = label_from_path(path.relative_to(root), keywords, priority)
        if label is None:
            unlabelled.append(path.relative_to(root).as_posix())
        else:
            labelled.append((path, label))
    return labelled, unlabelled


def rpm_of(path: Path, ev: dict[str, Any]) -> float | None:
    """Shaft speed parsed from the file name (evaluation.rpm_regex), e.g. 'Second speed - 1236rpm.mat' -> 1236."""
    m = re.search(ev.get("rpm_regex") or r"(\d{3,4})\s*rpm", path.name, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def condition_of(path: Path, ev: dict[str, Any]) -> str | None:
    """'<setup>-<rpm band>' parsed from the path, or None if the patterns don't match."""
    setup_re, rpm_re, bands = ev.get("setup_regex"), ev.get("rpm_regex"), ev.get("rpm_bands")
    if not (setup_re and rpm_re and bands):
        return None
    m_setup = re.search(setup_re, path.as_posix())
    m_rpm = re.search(rpm_re, path.name, flags=re.IGNORECASE)
    if not (m_setup and m_rpm):
        return None
    rpm = int(m_rpm.group(1))
    band = min(bands, key=lambda b: abs(b - rpm))
    return f"{m_setup.group(1)}-{band}"


# --------------------------------------------------------------------------- evaluation
def _training_matrix(recs: list[Recording], classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    blocks = [(r.features, r.label) for r in recs] + [(a, r.label) for r in recs for a in (r.augmented or [])]
    X = np.concatenate([b for b, _ in blocks])
    y = np.concatenate([[classes.index(lbl)] * len(b) for b, lbl in blocks])
    return X, y


def grouped_cv(recordings: list[Recording], classes: list[str], group_of, params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Hold out one group at a time; pool all held-out predictions into one report.

    Also returns out-of-fold recording-level class probabilities (``oof``) for conformal calibration.
    """
    groups = sorted({group_of(r) for r in recordings})
    y_true, y_pred, fold_acc, rec_hits, oof = [], [], [], [], []
    for held in groups:
        train = [r for r in recordings if group_of(r) != held]
        test = [r for r in recordings if group_of(r) == held]
        X_tr, y_tr = _training_matrix(train, classes)
        clf = lgb.LGBMClassifier(random_state=seed, **params).fit(X_tr, y_tr)
        fold_true, fold_pred = [], []
        for r in test:
            pred = clf.predict(r.features)
            truth = classes.index(r.label)
            fold_true += [truth] * len(pred)
            fold_pred += pred.tolist()
            # Recording-level decision: average the segment probabilities.
            mean_p = clf.predict_proba(r.features).mean(axis=0)
            full = np.zeros(len(classes))
            full[clf.classes_] = mean_p  # a class missing from this fold's training data gets probability 0
            rec_hits.append(int(full.argmax() == truth))
            oof.append((truth, full))
        fold_acc.append(accuracy_score(fold_true, fold_pred))
        y_true += fold_true
        y_pred += fold_pred
    labels = list(range(len(classes)))
    return {
        "folds": len(groups),
        "held_out_groups": groups,
        "segments": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_fold_std": float(np.std(fold_acc)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recording_accuracy": float(np.mean(rec_hits)) if rec_hits else None,
        "recordings": len(rec_hits),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0
        ),
        "oof": oof,
    }


def conformal_calibration(oof: list[tuple[int, np.ndarray]], alpha: float) -> dict[str, Any]:
    """Split-conformal quantile for prediction sets with score s = 1 - p(true class)."""
    scores = np.array([1.0 - p[t] for t, p in oof])
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(scores, level, method="higher"))
    sizes = [int(np.sum(1.0 - p <= q)) for _, p in oof]
    return {"alpha": alpha, "coverage": 1 - alpha, "q": q, "n_calibration": n, "mean_set_size": float(np.mean(sizes))}


def set_coverage(oof: list[tuple[int, np.ndarray]], q: float) -> dict[str, float]:
    hits = [int(1.0 - p[t] <= q) for t, p in oof]
    sizes = [int(np.sum(1.0 - p <= q)) for _, p in oof]
    return {"empirical_coverage": float(np.mean(hits)), "mean_set_size": float(np.mean(sizes))}


# --------------------------------------------------------------------------- training
def train_fault_model(files: list[tuple[Path, str]], cfg: dict[str, Any], out_dir: Path, unlabelled: list[str]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    fcfg = FaultFeatureConfig.from_dict(cfg["features"])
    ev = cfg.get("evaluation", {})
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings: list[Recording] = []
    for path, lbl in files:
        y = load_audio(path, fcfg.sample_rate, fcfg.channel, cfg.get("max_audio_seconds"))
        rpm = rpm_of(path, ev)
        if fcfg.order_bins and rpm is None:
            log.warning("No rpm in %s; order features will assume %.0f rpm", path.name, fcfg.default_rpm)
        try:
            feats = segment_features(y, fcfg, rpm)
        except ValueError:
            log.warning("Skipping short recording %s", path)
            continue
        augmented = None
        aug_cfg = cfg.get("augmentation") or {}
        if int(aug_cfg.get("copies", 0)) > 0:
            rng = np.random.default_rng([seed, len(recordings)])
            augmented = [
                segment_features(random_microphone(y, fcfg.sample_rate, rng, aug_cfg), fcfg, rpm)
                for _ in range(int(aug_cfg["copies"]))
            ]
        recordings.append(Recording(path, lbl, condition_of(path, ev), feats, augmented))
    classes = [c for c in cfg["classes"] if any(r.label == c for r in recordings)]
    params = {k: v for k, v in cfg["lightgbm"].items()}

    conditions = sorted({r.condition for r in recordings if r.condition})
    by_condition = len(conditions) >= 2 and all(r.condition for r in recordings)
    if by_condition:
        scheme = "leave_one_condition_out"
        cv = grouped_cv(recordings, classes, lambda r: r.condition, params, seed)
        setups = sorted({r.condition.split("-")[0] for r in recordings})
        cross_setup = (
            grouped_cv(recordings, classes, lambda r: r.condition.split("-")[0], params, seed) if len(setups) >= 2 else None
        )
    else:
        scheme = "leave_one_recording_out"
        cv = grouped_cv(recordings, classes, lambda r: r.path.as_posix(), params, seed)
        cross_setup = None
    log.info("CV (%s): accuracy=%.3f macro-F1=%.3f recording accuracy=%s", scheme, cv["accuracy"], cv["macro_f1"], cv["recording_accuracy"])
    if cross_setup:
        log.info("Cross-setup: accuracy=%.3f macro-F1=%.3f", cross_setup["accuracy"], cross_setup["macro_f1"])

    # Final model: everything except the demo hold-out condition.
    holdout = ev.get("demo_holdout") if by_condition and ev.get("demo_holdout") in conditions else None
    final = [r for r in recordings if r.condition != holdout] if holdout else recordings
    X, y = _training_matrix(final, classes)
    clf = lgb.LGBMClassifier(random_state=seed, **params).fit(X, y)

    alpha = float((cfg.get("conformal") or {}).get("alpha", 0.1))
    conformal = conformal_calibration(cv["oof"], alpha)
    if cross_setup:
        conformal["cross_setup"] = set_coverage(cross_setup["oof"], conformal["q"])
    log.info("Conformal sets: q=%.3f coverage target %.0f%% mean set size %.2f", conformal["q"], 100 * (1 - alpha), conformal["mean_set_size"])
    clf.booster_.save_model(str(out_dir / "model.txt"))

    demo_pool = [r for r in recordings if r.condition == holdout] if holdout else recordings
    samples = _export_samples(demo_pool, fcfg, cfg.get("max_audio_seconds"))

    caveats = [
        "Lab test rig (Gunt PT 500, SKF 6004) with artificial defects; real machines will differ.",
        "Headline metrics: leave-one-condition-out CV (unseen speed + recording setup combination).",
        "Features ignore loudness (spectral shape, envelope spectrum, crest factor, kurtosis, ...), and training adds "
        "simulated-microphone copies, so volume and mic colouring matter less.",
        f"Prediction sets aim to contain the true class {1 - alpha:.0%} of the time for recordings like the training data; "
        "coverage can drop for very different recording setups.",
    ]
    if cross_setup:
        caveats.append(
            f"A different microphone setup is much harder: leave-one-setup-out accuracy {cross_setup['accuracy']:.0%}. "
            "Record new machines the same way as the training data, or retrain with their recordings."
        )
    caveats.append(
        f"Demo clips come from condition '{holdout}', which the final model never saw."
        if holdout
        else "Demo clips were part of the training data, so predictions on them are optimistic."
    )

    metadata = {
        "kind": "bearing_fault",
        "version": __version__,
        "created_at": utc_now(),
        "dataset": "mendeley_n9y9c7xrz3",
        "model_file": "model.txt",
        "features": fcfg.to_dict(),
        "feature_names": feature_names(fcfg),
        "classes": classes,
        "class_labels": {c: cfg["classes"][c]["label"] for c in classes},
        "max_audio_seconds": cfg.get("max_audio_seconds"),
        "split_strategy": scheme,
        "conditions": conditions,
        "demo_holdout": holdout,
        "final_model_recordings": len(final),
        "label_counts": dict(Counter(r.label for r in recordings)),
        "unlabelled_files": unlabelled[:200],
        "demo_samples": samples,
        "augmentation": cfg.get("augmentation") or {},
        "conformal": conformal,
        "test_metrics": {
            **{k: v for k, v in cv.items() if k != "oof"},
            "scheme": scheme,
            "cross_setup": {k: v for k, v in cross_setup.items() if k != "oof"} if cross_setup else None,
        },
        "caveats": caveats,
    }
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def _export_samples(recordings: list[Recording], fcfg: FaultFeatureConfig, max_seconds) -> list[str]:
    """Write one recording per class to data/samples/bearing_<class>_0.wav for the UI's sample picker."""
    import soundfile as sf

    out = samples_dir()
    out.mkdir(parents=True, exist_ok=True)
    written, seen = [], set()
    for r in recordings:
        if r.label in seen:
            continue
        y = load_audio(r.path, fcfg.sample_rate, fcfg.channel, max_seconds or 15)
        name = f"bearing_{r.label}_0.wav"
        sf.write(out / name, np.clip(y, -1.0, 1.0), fcfg.sample_rate, subtype="PCM_16")
        written.append(name)
        seen.add(r.label)
    log.info("Wrote %d demo clips to %s", len(written), out)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(default_config("bearing")))
    parser.add_argument("--data-root")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(args.data_root or cfg.get("data_root") or raw_dir() / "bearing_sound")
    if not root.is_dir() or not any(p.suffix.lower() in AUDIO_EXT for p in root.rglob("*")):
        from ..datasets.download import get_bearing

        root = get_bearing()  # first run: fetch the 75 MB dataset automatically
    files, unlabelled = discover_labelled(root, cfg["classes"])
    if unlabelled:
        log.warning("%d audio files matched no (or several) classes, e.g. %s", len(unlabelled), unlabelled[:5])
    if len({lbl for _, lbl in files}) < 2:
        raise SystemExit("Need at least two classes. Check `classes.*.keywords` in configs/bearing.yaml.")

    out_dir = Path(args.out) if args.out else models_dir() / cfg.get("output_subdir", "bearing")
    meta = train_fault_model(files, cfg, out_dir, unlabelled)
    m = meta["test_metrics"]
    print(f"\nSaved bearing classifier to {out_dir}")
    print(f"Label counts: {meta['label_counts']}")
    print(f"Evaluation ({m['scheme']}): accuracy={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}  "
          f"recording accuracy={m['recording_accuracy']:.3f}  (chance ~{1 / len(meta['classes']):.3f})")
    if m.get("cross_setup"):
        print(f"Different recording setup: accuracy={m['cross_setup']['accuracy']:.3f}")
    c = meta["conformal"]
    print(f"Prediction sets: target coverage {c['coverage']:.0%}, average size {c['mean_set_size']:.2f} classes"
          + (f", coverage on a different mic setup {c['cross_setup']['empirical_coverage']:.0%}" if c.get("cross_setup") else ""))
    print(f"Demo clips: {meta['demo_samples']}")


if __name__ == "__main__":
    main()
