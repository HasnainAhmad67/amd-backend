"""Train the sensor failure-risk models.

    telltale-train-sensor                       # real AI4I data (run `telltale-download ai4i` first)
    telltale-train-sensor --synthetic           # AI4I-style synthetic data (smoke test / offline)

Artifacts written to <models>/sensor/:
    failure.txt          LightGBM booster: P(any machine failure)
    mode_<CODE>.txt      LightGBM booster per failure mode (TWF, HDF, PWF, OSF)
    metadata.json        features, thresholds, feature ranges, test metrics, provenance
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

from .. import __version__
from ..metrics import best_f1_threshold, binary_report
from ..paths import default_config, models_dir, processed_dir
from ..utils import get_logger, load_config, seed_everything, utc_now, write_json
from .data import default_ai4i_path, load_ai4i
from .features import engineer
from .schema import FAILURE_MODES, FEATURE_LABELS, FEATURES, SENSOR_FIELDS, TARGET
from .synthetic import generate

log = get_logger("telltale.sensor.train")

# LightGBM >= 4.6 renamed eval_set -> eval_X/eval_y; we keep eval_set so older 4.x versions still work.
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")


def _split(df: pd.DataFrame, cfg: dict[str, Any], seed: int):
    test_frac = cfg["data"]["test_fraction"]
    val_frac = cfg["data"]["val_fraction"]
    train_val, test = train_test_split(df, test_size=test_frac, stratify=df[TARGET], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_frac / (1 - test_frac), stratify=train_val[TARGET], random_state=seed
    )
    return train, val, test


def _fit_binary(X_tr, y_tr, X_val, y_val, cfg: dict[str, Any], seed: int) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(random_state=seed, **cfg["lightgbm"])
    callbacks = []
    if 0 < y_val.sum() < len(y_val):
        callbacks.append(lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False))
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
    return clf


def _feature_ranges(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "p01": float(X[col].quantile(0.01)),
            "p50": float(X[col].quantile(0.50)),
            "p99": float(X[col].quantile(0.99)),
        }
        for col in X.columns
    }


def train_sensor_models(df: pd.DataFrame, cfg: dict[str, Any], out_dir: Path, source: str) -> dict[str, Any]:
    """Train + evaluate + save. ``df`` must have the canonical AI4I columns (see schema.AI4I_COLUMN_MAP)."""
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = _split(df, cfg, seed)
    X_tr, X_val, X_te = engineer(train), engineer(val), engineer(test)
    log.info("Split sizes: train=%d val=%d test=%d (failure prevalence %.3f)", len(train), len(val), len(test), df[TARGET].mean())

    models_meta: dict[str, Any] = {"modes": {}}

    # Overall failure model.
    clf = _fit_binary(X_tr, train[TARGET], X_val, val[TARGET], cfg, seed)
    threshold = best_f1_threshold(val[TARGET].to_numpy(), clf.predict_proba(X_val)[:, 1])
    test_report = binary_report(test[TARGET].to_numpy(), clf.predict_proba(X_te)[:, 1], threshold)
    clf.booster_.save_model(str(out_dir / "failure.txt"))
    models_meta["failure"] = {"file": "failure.txt", "threshold": threshold, "test_metrics": test_report}
    log.info("failure: threshold=%.3f test PR-AUC=%s ROC-AUC=%s F1=%.3f", threshold, test_report["pr_auc"], test_report["roc_auc"], test_report["f1"])

    # One model per failure mode (multi-label: a row can have several modes).
    for code, name in FAILURE_MODES.items():
        if train[code].sum() < 3:
            log.warning("Skipping %s: only %d positives in train split", code, int(train[code].sum()))
            continue
        mclf = _fit_binary(X_tr, train[code], X_val, val[code], cfg, seed)
        val_proba = mclf.predict_proba(X_val)[:, 1]
        mthr = best_f1_threshold(val[code].to_numpy(), val_proba)
        val_f1 = binary_report(val[code].to_numpy(), val_proba, mthr)["f1"]
        reliable = val_f1 >= float(cfg.get("min_mode_val_f1", 0.5))
        mreport = binary_report(test[code].to_numpy(), mclf.predict_proba(X_te)[:, 1], mthr)
        fname = f"mode_{code}.txt"
        mclf.booster_.save_model(str(out_dir / fname))
        models_meta["modes"][code] = {
            "file": fname, "name": name, "threshold": mthr,
            "val_f1": val_f1, "reliable": reliable, "test_metrics": mreport,
        }
        log.info("%s: threshold=%.3f val F1=%.3f reliable=%s | test PR-AUC=%s F1=%.3f", code, mthr, val_f1, reliable, mreport["pr_auc"], mreport["f1"])

    importance = dict(zip(FEATURES, clf.booster_.feature_importance(importance_type="gain").tolist(), strict=True))
    medium_fraction = float(cfg["risk"]["medium_fraction"])
    metadata = {
        "kind": "sensor",
        "version": __version__,
        "created_at": utc_now(),
        "data_source": source,
        "n_rows": int(len(df)),
        "split": {"train": len(train), "val": len(val), "test": len(test), "seed": seed},
        "features": list(FEATURES),
        "feature_labels": FEATURE_LABELS,
        "sensor_fields": list(SENSOR_FIELDS),
        "feature_ranges": _feature_ranges(X_tr),
        "feature_importance_gain": importance,
        "risk_thresholds": {"medium": threshold * medium_fraction, "high": threshold},
        "models": models_meta,
        "caveats": [
            "AI4I 2020 is a synthetic dataset whose labels follow published rules; scores on it overstate real-world performance.",
            "Contributions are SHAP values in log-odds space (LightGBM pred_contrib); they explain the model, not physics.",
            "Random failures (RNF) are not modelled because they are random by construction.",
            "Failure-mode models with validation F1 below min_mode_val_f1 are shown as low-confidence and never raise flags.",
        ],
    }
    if source.startswith("synthetic"):
        metadata["caveats"].insert(
            0,
            "Trained on SIMULATED data (the real AI4I dataset was not available). Fine for a demo; "
            "retrain with real data before quoting any numbers.",
        )
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def train_default(out_dir: Path, offline: bool = False) -> dict[str, Any]:
    """Train with the bundled config: real AI4I if it can be downloaded, otherwise synthetic data.

    Used by the API to build the sensor model automatically on first start.
    """
    cfg = load_config(default_config("sensor"))
    if not offline:
        try:
            from ..datasets.download import get_ai4i

            return train_sensor_models(load_ai4i(get_ai4i(timeout=30)), cfg, out_dir, "ai4i2020")
        except Exception as exc:  # no internet, blocked URL, ...: still give the demo a working model
            log.warning("Could not get the AI4I dataset (%s); training on synthetic AI4I-style data instead.", exc)
    return train_sensor_models(generate(10_000, seed=int(cfg.get("seed", 42))), cfg, out_dir, "synthetic-ai4i-rules")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(default_config("sensor")))
    parser.add_argument("--data", default=None, help="Path to ai4i2020.csv (overrides config)")
    parser.add_argument("--synthetic", action="store_true", help="Train on AI4I-style synthetic data")
    parser.add_argument("--out", default=None, help="Output dir (default: <models>/sensor)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.synthetic:
        df, source = generate(10_000, seed=int(cfg.get("seed", 42))), "synthetic-ai4i-rules"
    else:
        path = args.data or cfg["data"].get("path")
        if path is None and not default_ai4i_path().is_file():
            from ..datasets.download import get_ai4i

            path = get_ai4i()  # first run: fetch the 0.5 MB dataset automatically
        df, source = load_ai4i(path), "ai4i2020"

    out_dir = Path(args.out) if args.out else models_dir() / cfg.get("output_subdir", "sensor")
    meta = train_sensor_models(df, cfg, out_dir, source)

    # Keep the held-out rows around for demos / notebooks.
    processed_dir().mkdir(parents=True, exist_ok=True)
    _, _, test = _split(df, cfg, int(cfg.get("seed", 42)))
    test.to_csv(processed_dir() / "sensor_test_split.csv", index=False)

    f = meta["models"]["failure"]["test_metrics"]
    print(f"\nSaved sensor models to {out_dir}")
    print(f"Test: PR-AUC={f['pr_auc']}  ROC-AUC={f['roc_auc']}  F1={f['f1']:.3f}  (prevalence {f['prevalence']:.3f})")


if __name__ == "__main__":
    main()
