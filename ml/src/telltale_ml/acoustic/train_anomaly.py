"""Train a machine-sound anomaly detector (one per machine type) and export it to ONNX.

    telltale-train-acoustic --machine-type fan
    telltale-train-acoustic --dataset mimii --machine-type pump --data-root data/raw/mimii/6dB
    telltale-train-acoustic --machine-type fan --epochs 5 --max-train-clips 300 --n-models 1   # quick run

Two methods (configs/acoustic.yaml -> method):
    ensemble     (default) CNN embeddings trained with an ArcFace loss + classical spectra, nearest-neighbour
                 scoring, conformal p-values. See train_ensemble.py.
    autoencoder  the DCASE 2020/21 baseline: reconstruction error of stacked log-mel frames.

Runs on an AMD GPU automatically inside the ROCm image (torch.cuda == HIP there).

Artifacts written to <models>/acoustic/<machine_type>/:
    model.onnx       features (N, D) -> vector_error (N,)   [normalisation baked in]
    metadata.json    feature config, threshold, reference spectrum, test metrics
"""

from __future__ import annotations

import argparse
import contextlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .. import __version__
from ..metrics import anomaly_report
from ..paths import default_config, models_dir, raw_dir
from ..utils import get_logger, load_config, seed_everything, utc_now, write_json
from .data import Clip, discover
from .features import LogMelConfig, load_audio, log_mel, mel_band_hz, stack_frames
from .model import DenseAutoencoder, ScoringModel, pick_device

log = get_logger("telltale.acoustic.train")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_):  # type: ignore[no-redef]
        return it


# --------------------------------------------------------------------------- feature jobs
# Top-level functions so they can run in worker processes.
def _train_job(args: tuple[str, dict, int, int]) -> tuple[np.ndarray, np.ndarray]:
    path, cfg_dict, per_clip, seed = args
    cfg = LogMelConfig.from_dict(cfg_dict)
    lm = log_mel(load_audio(path, cfg.sample_rate, cfg.channel), cfg)
    vecs = stack_frames(lm, cfg.frames)
    if per_clip and len(vecs) > per_clip:
        idx = np.random.default_rng(seed).choice(len(vecs), per_clip, replace=False)
        vecs = vecs[idx]
    return vecs, lm.mean(axis=1)


def _vectors_job(args: tuple[str, dict]) -> np.ndarray:
    path, cfg_dict = args
    cfg = LogMelConfig.from_dict(cfg_dict)
    return stack_frames(log_mel(load_audio(path, cfg.sample_rate, cfg.channel), cfg), cfg.frames)


def _pool(workers: int):
    return ProcessPoolExecutor(workers) if workers > 1 else contextlib.nullcontext()


def _map(executor, fn, jobs):
    return executor.map(fn, jobs, chunksize=4) if executor is not None else map(fn, jobs)


# --------------------------------------------------------------------------- training
def _score_clips(scorer: ScoringModel, clips: list[Clip], cfg: LogMelConfig, device, workers: int, desc: str) -> np.ndarray:
    scores = []
    scorer.eval()
    with _pool(workers) as ex, torch.no_grad():
        jobs = [(str(c.path), cfg.to_dict()) for c in clips]
        for vecs in tqdm(_map(ex, _vectors_job, jobs), total=len(jobs), desc=desc):
            err = scorer(torch.from_numpy(vecs).to(device))
            scores.append(float(err.mean().item()))
    return np.asarray(scores)


def export_onnx(scorer: ScoringModel, input_dim: int, path: Path) -> None:
    scorer = scorer.cpu().eval()
    dummy = torch.zeros(2, input_dim, dtype=torch.float32)
    kwargs: dict[str, Any] = dict(
        input_names=["features"],
        output_names=["vector_error"],
        dynamic_axes={"features": {0: "n"}, "vector_error": {0: "n"}},
        opset_version=17,
    )
    try:  # torch >= 2.5: force the stable TorchScript exporter
        torch.onnx.export(scorer, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(scorer, (dummy,), str(path), **kwargs)

    # Verify the exported graph matches torch.
    import onnxruntime as ort

    sample = torch.randn(16, input_dim) * 10 - 40
    ref = scorer(sample).detach().numpy()
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"features": sample.numpy()})[0]
    if not np.allclose(ref, got, rtol=1e-3, atol=1e-4):
        raise RuntimeError(f"ONNX export mismatch: max abs diff {np.abs(ref - got).max()}")


def train_anomaly_model(clips: list[Clip], cfg: dict[str, Any], out_dir: Path, dataset: str, machine_type: str) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    tcfg, mcfg = cfg["train"], cfg["model"]
    feat_cfg = LogMelConfig.from_dict(cfg["features"])
    device = pick_device(tcfg.get("device", "auto"))
    workers = int(tcfg.get("num_workers", 1))
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_train = [c for c in clips if c.split == "train" and c.label == "normal"]
    test = [c for c in clips if c.split == "test"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(normal_train))
    n_val = max(1, int(len(normal_train) * tcfg["val_fraction"]))
    val = [normal_train[i] for i in order[:n_val]]
    fit = [normal_train[i] for i in order[n_val:]]
    if tcfg.get("max_train_clips"):
        fit = fit[: int(tcfg["max_train_clips"])]
    log.info("Clips: fit=%d val(normal)=%d test=%d | device=%s", len(fit), len(val), len(test), device)

    # 1) Features for the fit set.
    vec_chunks, band_means = [], []
    with _pool(workers) as ex:
        jobs = [(str(c.path), feat_cfg.to_dict(), int(tcfg["vectors_per_clip"]), seed + i) for i, c in enumerate(fit)]
        for vecs, bands in tqdm(_map(ex, _train_job, jobs), total=len(jobs), desc="features"):
            vec_chunks.append(vecs)
            band_means.append(bands)
    X = np.concatenate(vec_chunks).astype(np.float32)
    band_means_arr = np.stack(band_means)
    mean, std = X.mean(axis=0), X.std(axis=0)

    # 2) Train.
    ae = DenseAutoencoder(feat_cfg.input_dim, mcfg["hidden"], mcfg["bottleneck"], mcfg["depth"])
    scorer = ScoringModel(ae, mean, std).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=float(tcfg["learning_rate"]))
    batch = min(int(tcfg["batch_size"]), len(X))
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch, shuffle=True, drop_last=True)
    losses = []
    for epoch in range(int(tcfg["epochs"])):
        scorer.train()
        running = 0.0
        for (xb,) in loader:
            loss = scorer(xb.to(device, non_blocking=True)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
        losses.append(running / max(1, len(loader)))
        log.info("epoch %d/%d loss %.4f", epoch + 1, tcfg["epochs"], losses[-1])

    # 3) Threshold from held-out normal clips.
    val_scores = _score_clips(scorer, val, feat_cfg, device, workers, "val")
    threshold = float(np.percentile(val_scores, cfg["threshold_percentile"]))

    # 4) Evaluate on the official / held-out test set.
    test_scores = _score_clips(scorer, test, feat_cfg, device, workers, "test")
    y = np.array([c.label == "anomaly" for c in test], dtype=int)
    metrics: dict[str, Any] = {"overall": anomaly_report(y, test_scores, threshold)}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(test):
        groups[f"{c.group}/{c.domain}"].append(i)
    metrics["by_group"] = {g: anomaly_report(y[idx], test_scores[idx], threshold) for g, idx in sorted(groups.items())}
    log.info("Test AUC=%s pAUC=%s", metrics["overall"]["roc_auc"], metrics["overall"]["pauc"])

    # 5) Export.
    export_onnx(scorer, feat_cfg.input_dim, out_dir / "model.onnx")
    metadata = {
        "kind": "acoustic_anomaly",
        "version": __version__,
        "created_at": utc_now(),
        "dataset": dataset,
        "machine_type": machine_type,
        "features": feat_cfg.to_dict(),
        "input_dim": feat_cfg.input_dim,
        "onnx": {"file": "model.onnx", "input": "features", "output": "vector_error"},
        "threshold": threshold,
        "threshold_percentile": cfg["threshold_percentile"],
        "watch_fraction": cfg["watch_fraction"],
        "max_audio_seconds": cfg["max_audio_seconds"],
        "val_normal_scores": {
            "p50": float(np.percentile(val_scores, 50)),
            "p95": float(np.percentile(val_scores, 95)),
            "max": float(val_scores.max()),
        },
        "reference_spectrum": {
            "band_hz": mel_band_hz(feat_cfg),
            "mean": band_means_arr.mean(axis=0),
            "std": band_means_arr.std(axis=0),
        },
        "test_metrics": metrics,
        "training": {
            "device": str(device),
            "epochs": int(tcfg["epochs"]),
            "fit_clips": len(fit),
            "vectors": int(len(X)),
            "loss_curve": losses,
        },
        "caveats": [
            "Trained on normal sounds only; it flags 'different from normal', not a specific fault.",
            "Threshold is a percentile of held-out normal scores; expect that share of normal clips above it to be ~"
            f"{100 - cfg['threshold_percentile']}%.",
            "Recording setup (mic, distance, background noise) must resemble the training data.",
        ],
    }
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(default_config("acoustic")))
    parser.add_argument("--dataset", choices=["mimii_due", "mimii"])
    parser.add_argument("--machine-type")
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-clips", type=int)
    parser.add_argument("--n-models", type=int, help="ensemble: number of CNNs to train and average")
    parser.add_argument("--device")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dataset = args.dataset or cfg["dataset"]
    machine_type = args.machine_type or cfg["machine_type"]
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.max_train_clips:
        cfg["train"]["max_train_clips"] = args.max_train_clips
    if args.n_models:
        cfg.setdefault("ensemble", {})["n_models"] = args.n_models
    if args.device:
        cfg["train"]["device"] = args.device

    root = Path(args.data_root or cfg.get("data_root") or raw_dir() / dataset)
    clips = discover(dataset, root, machine_type, seed=int(cfg.get("seed", 42)))
    out_dir = Path(args.out) if args.out else models_dir() / cfg.get("output_subdir", "acoustic") / machine_type
    if cfg.get("method", "ensemble") == "ensemble":
        from .train_ensemble import train_ensemble_model

        meta = train_ensemble_model(clips, cfg, out_dir, dataset, machine_type)
        t = meta["test_metrics"]
        print(f"\nSaved {machine_type} detector (ensemble) to {out_dir}")
        print(f"Test, section unknown: DCASE hmean={t['dcase_hmean']:.3f}  AUC source={t['mean_auc_source']:.3f}  "
              f"target={t['mean_auc_target']:.3f}  pAUC={t['overall']['pauc']:.3f}")
        return
    meta = train_anomaly_model(clips, cfg, out_dir, dataset, machine_type)
    m = meta["test_metrics"]["overall"]
    print(f"\nSaved {machine_type} detector to {out_dir}")
    print(f"Test AUC={m['roc_auc']}  pAUC={m['pauc']}  threshold={meta['threshold']:.3f}")


if __name__ == "__main__":
    main()
