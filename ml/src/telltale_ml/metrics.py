"""Evaluation helpers. Report honest metrics: imbalanced data => PR-AUC + prevalence, not accuracy."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray, default: float = 0.5) -> float:
    """Threshold maximising F1 on a validation set (falls back to ``default`` if no positives)."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return default
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision_recall_curve returns len(thresholds) + 1 points; drop the last one.
    idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[idx])


def binary_report(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    pred = (proba >= threshold).astype(int)
    both_classes = 0 < y_true.sum() < len(y_true)
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(y_true) else 0.0,
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, proba)) if both_classes else None,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def anomaly_report(y_true: np.ndarray, scores: np.ndarray, threshold: float, max_fpr: float = 0.1) -> dict[str, Any]:
    """AUC / pAUC for anomaly scores (higher = more anomalous). pAUC is sklearn's McClish-standardised value."""
    y_true = np.asarray(y_true).astype(int)
    both_classes = 0 < y_true.sum() < len(y_true)
    report = binary_report(y_true, scores, threshold)
    report.pop("pr_auc")
    report["roc_auc"] = float(roc_auc_score(y_true, scores)) if both_classes else None
    report["pauc"] = float(roc_auc_score(y_true, scores, max_fpr=max_fpr)) if both_classes else None
    report["pauc_max_fpr"] = max_fpr
    return report
