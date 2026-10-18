"""Sound anomaly detector v2 ("ensemble"): CNN embeddings (ArcFace) + classical spectra, nearest-neighbour scoring.

Called by `telltale-train-acoustic` when `method: ensemble` (the default in configs/acoustic.yaml).

Pipeline
  1. Log-mel spectrograms for every clip.
  2. Hold out `calibration_fraction` of the normal SOURCE training clips per section: never trained on, never put in
     a bank - their scores calibrate the threshold and the conformal p-values.
  3. Train `n_models` CNNs (different seeds) to classify section x domain with an ArcFace loss -> ONNX each.
     Averaging several runs matters: one CNN's quality varies a lot between seeds.
  4. Banks of normal embeddings / spectra per (section, domain) + leave-one-out normalisation statistics.
  5. Score the official test clips WITHOUT telling the scorer their section (exactly how the API scores uploads)
     and report per-section/domain AUC, pAUC and the DCASE-style harmonic mean.

Artifacts in <models>/acoustic/<machine_type>/: embedder_<k>.onnx, bank.npz, metadata.json
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import harmonic_mean
from typing import Any

import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_auc_score

from .. import __version__
from ..metrics import anomaly_report
from ..utils import get_logger, seed_everything, utc_now, write_json
from .data import Clip
from .embedding import export_embedder, train_embedder
from .features import LogMelConfig, load_audio, log_mel, mel_band_hz
from .knn_scorer import EnsembleScorer, l2_normalize, leave_one_out_stats
from .model import pick_device

log = get_logger("telltale.acoustic.ensemble")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_):  # type: ignore[no-redef]
        return it


def _logmel_job(args: tuple[str, dict]) -> np.ndarray:
    path, cfg_dict = args
    cfg = LogMelConfig.from_dict(cfg_dict)
    return log_mel(load_audio(path, cfg.sample_rate, cfg.channel), cfg).astype(np.float16)


def compute_logmels(clips: list[Clip], cfg: LogMelConfig, workers: int) -> list[np.ndarray]:
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(str(c.path), cfg.to_dict()) for c in clips]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            return list(tqdm(ex.map(_logmel_job, jobs, chunksize=8), total=len(jobs), desc="spectrograms"))
    return [_logmel_job(j) for j in tqdm(jobs, desc="spectrograms")]


def embed(session: ort.InferenceSession, specs: list[np.ndarray]) -> np.ndarray:
    """ONNX embeddings, batching clips of equal length."""
    out = np.zeros((len(specs), session.get_outputs()[0].shape[1] or 128), dtype=np.float32)
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(specs):
        by_len[s.shape[1]].append(i)
    for idx in by_len.values():
        for start in range(0, len(idx), 64):
            chunk = idx[start : start + 64]
            batch = np.stack([specs[i] for i in chunk]).astype(np.float32)
            out[chunk] = session.run(None, {"logmel": batch})[0]
    return out


def classical_vectors(specs: list[np.ndarray]) -> np.ndarray:
    return np.stack([s.astype(np.float32).mean(axis=1) for s in specs])


def dcase_summary(clips: list[Clip], scores: np.ndarray, max_fpr: float = 0.1) -> dict[str, Any]:
    """AUC/pAUC per (section, domain) + harmonic mean over all of them (DCASE official-style score)."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, c in enumerate(clips):
        groups[(c.group, c.domain)].append(i)
    rows, values, src, tgt = {}, [], [], []
    for (group, domain), idx in sorted(groups.items()):
        y = np.array([clips[i].label == "anomaly" for i in idx], dtype=int)
        if 0 < y.sum() < len(y):
            auc = float(roc_auc_score(y, scores[idx]))
            pauc = float(roc_auc_score(y, scores[idx], max_fpr=max_fpr))
            rows[f"{group}/{domain}"] = {"roc_auc": auc, "pauc": pauc, "n": len(idx)}
            values += [auc, pauc]
            (src if domain == "source" else tgt).append(auc)
    return {
        "hmean": float(harmonic_mean(values)) if values else None,
        "mean_auc_source": float(np.mean(src)) if src else None,
        "mean_auc_target": float(np.mean(tgt)) if tgt else None,
        "by_group": rows,
    }


def train_ensemble_model(clips: list[Clip], cfg: dict[str, Any], out_dir: Path, dataset: str, machine_type: str) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    tcfg, ecfg = cfg["train"], cfg.get("ensemble", {})
    feat_cfg = LogMelConfig.from_dict(cfg["features"])
    device = pick_device(tcfg.get("device", "auto"))
    workers = int(tcfg.get("num_workers", 1))
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_train = [c for c in clips if c.split == "train" and c.label == "normal"]
    test = [c for c in clips if c.split == "test"]
    # By default only use sections that have labelled test clips: the metrics then describe exactly the
    # deployed model. (Tested: adding MIMII DUE's extra training-only sections made fan results WORSE.)
    if ecfg.get("only_evaluated_groups", True) and test:
        evaluated = {c.group for c in test}
        skipped = sorted({c.group for c in normal_train} - evaluated)
        if skipped:
            log.info("Ignoring training-only sections %s (ensemble.only_evaluated_groups=true)", skipped)
        normal_train = [c for c in normal_train if c.group in evaluated]

    # 2) Calibration hold-out: a slice of each section's normal SOURCE clips.
    cal_frac = float(ecfg.get("calibration_fraction", 0.1))
    fit, cal = [], []
    by_group: dict[str, list[Clip]] = defaultdict(list)
    for c in normal_train:
        (by_group[c.group] if c.domain == "source" else fit).append(c)
    for _group, items in sorted(by_group.items()):
        order = rng.permutation(len(items))
        n_cal = max(1, int(len(items) * cal_frac)) if len(items) > 5 else 0
        cal += [items[i] for i in order[:n_cal]]
        fit += [items[i] for i in order[n_cal:]]
    if tcfg.get("max_train_clips"):
        fit = [fit[i] for i in rng.permutation(len(fit))[: int(tcfg["max_train_clips"])]]
    log.info("Clips: fit=%d calibration=%d test=%d | device=%s", len(fit), len(cal), len(test), device)

    # 1) Spectrograms.
    all_clips = fit + cal + test
    specs = compute_logmels(all_clips, feat_cfg, workers)
    s_fit, s_cal, s_test = specs[: len(fit)], specs[len(fit) : len(fit) + len(cal)], specs[len(fit) + len(cal) :]

    # 3) CNN embedder on condition labels.
    t_min = min(s.shape[1] for s in s_fit)
    train_specs = np.stack([s[:, :t_min] for s in s_fit]).astype(np.float32)
    label_names = sorted({f"{c.group}__{c.domain}" for c in fit})
    labels = np.array([label_names.index(f"{c.group}__{c.domain}") for c in fit])
    emb_cfg = {**ecfg, "epochs": tcfg.get("epochs", 20), "batch_size": ecfg.get("batch_size", 64)}
    n_models = int(ecfg.get("n_models", 1))
    for stale in out_dir.glob("embedder*.onnx"):
        stale.unlink()
    embedders, views_fit, views_cal, views_test = [], {}, {}, {}
    for k in range(n_models):
        log.info("Training CNN %d/%d", k + 1, n_models)
        net = train_embedder(train_specs, labels, emb_cfg, device, seed + 1000 * k)
        onnx_path = out_dir / f"embedder_{k}.onnx"
        export_embedder(net, feat_cfg.n_mels, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        branch = f"arc{k}"
        embedders.append({"branch": branch, "file": onnx_path.name, "input": "logmel", "output": "embedding"})
        # 4) Banks + normalisation statistics use the deployment code path: ONNX embeddings.
        views_fit[branch], views_cal[branch], views_test[branch] = (l2_normalize(embed(session, s)) for s in (s_fit, s_cal, s_test))

    cls_raw = classical_vectors(s_fit)
    cls_mu, cls_sd = cls_raw.mean(axis=0), cls_raw.std(axis=0) + 1e-6
    standardize = lambda v: (v - cls_mu) / cls_sd  # noqa: E731
    views_fit["cls"], views_cal["cls"], views_test["cls"] = (
        standardize(cls_raw), standardize(classical_vectors(s_cal)), standardize(classical_vectors(s_test))
    )

    banks: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, list[float]]] = {b: {} for b in views_fit}
    groups = sorted({c.group for c in fit})
    for branch, vecs in views_fit.items():
        for group in groups:
            for domain in ("source", "target"):
                idx = [i for i, c in enumerate(fit) if c.group == group and c.domain == domain]
                if idx:
                    banks[f"{branch}__{group}__{domain}"] = vecs[idx].astype(np.float32)
            src_idx = [i for i, c in enumerate(fit) if c.group == group and c.domain == "source"] or [
                i for i, c in enumerate(fit) if c.group == group
            ]
            stats[branch][group] = list(leave_one_out_stats(branch, vecs[src_idx]))
    np.savez_compressed(out_dir / "bank.npz", **banks)
    weights = {"arc": float(ecfg.get("weight_arc", 0.5)), "cls": float(ecfg.get("weight_cls", 0.5))}
    scorer = EnsembleScorer(banks, stats, weights)

    # Calibration scores -> threshold + conformal p-values.
    cal_scores = scorer.score(views_cal) if len(cal) else np.array([0.0])
    threshold = float(np.percentile(cal_scores, cfg["threshold_percentile"]))

    # 5) Evaluation, section unknown (exactly like the API).
    test_scores = scorer.score(views_test)
    y = np.array([c.label == "anomaly" for c in test], dtype=int)
    summary = dcase_summary(test, test_scores)
    metrics: dict[str, Any] = {
        "overall": anomaly_report(y, test_scores, threshold) if len(test) else {},
        "by_group": summary["by_group"],
        "dcase_hmean": summary["hmean"],
        "mean_auc_source": summary["mean_auc_source"],
        "mean_auc_target": summary["mean_auc_target"],
        "section_known": False,
    }
    log.info(
        "Test (section unknown): hmean=%s source AUC=%s target AUC=%s",
        summary["hmean"], summary["mean_auc_source"], summary["mean_auc_target"],
    )

    band_means = np.stack([s.astype(np.float32).mean(axis=1) for s in s_fit])
    alpha = 1 - float(cfg["threshold_percentile"]) / 100
    metadata = {
        "kind": "acoustic_anomaly",
        "scorer": "ensemble",
        "version": __version__,
        "created_at": utc_now(),
        "dataset": dataset,
        "machine_type": machine_type,
        "features": feat_cfg.to_dict(),
        "embedders": embedders,
        "bank_file": "bank.npz",
        "classical": {"mean": cls_mu, "std": cls_sd},
        "branch_stats": stats,
        "weights": weights,
        "conditions": label_names,
        "threshold": threshold,
        "threshold_percentile": cfg["threshold_percentile"],
        "alpha": alpha,
        "watch_alpha": float(ecfg.get("watch_alpha", 0.15)),
        "watch_fraction": cfg["watch_fraction"],
        "max_audio_seconds": cfg["max_audio_seconds"],
        "window_frames": int(ecfg.get("window_frames", 64)),
        "window_hop_frames": int(ecfg.get("window_hop_frames", 16)),
        "calibration_scores": np.sort(cal_scores),
        "val_normal_scores": {
            "p50": float(np.percentile(cal_scores, 50)),
            "p95": float(np.percentile(cal_scores, 95)),
            "max": float(cal_scores.max()),
        },
        "reference_spectrum": {"band_hz": mel_band_hz(feat_cfg), "mean": band_means.mean(axis=0), "std": band_means.std(axis=0)},
        "test_metrics": metrics,
        "training": {
            "device": str(device), "epochs": int(tcfg.get("epochs", 20)), "n_models": n_models,
            "fit_clips": len(fit), "calibration_clips": len(cal),
        },
        "caveats": [
            "Trained on normal sounds only; it flags 'different from every normal condition it knows', not a specific fault.",
            "Metrics are measured without telling the model each clip's section - the same way uploads are scored.",
            f"p-values compare a clip with {len(cal)} held-out normal clips: p = 0.05 means only 5% of normal clips look this unusual.",
            "Recording setup (mic, distance, background noise) must resemble the training data.",
        ],
    }
    write_json(out_dir / "metadata.json", metadata)
    # The ensemble replaces the autoencoder; drop a stale model.onnx so nobody loads the wrong one.
    (out_dir / "model.onnx").unlink(missing_ok=True)
    return metadata
