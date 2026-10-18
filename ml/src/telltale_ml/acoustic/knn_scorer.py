"""Nearest-neighbour anomaly scoring shared by training and the API (torch-free).

Complementary views of a clip are compared with banks of NORMAL training clips:
  * "arc0", "arc1", ...: embeddings of independently trained CNNs (cosine distance). Several seeds are averaged
    because a single CNN's quality varies a lot from run to run.
  * "cls":  classical summary - the clip's average log-mel spectrum, standardised (euclidean distance)

For every (section, domain) bank we take the nearest-neighbour distance, z-normalise it with that section's
leave-one-out distances between normal clips, and keep the minimum over all banks - so the clip's section and
domain do not need to be known (a clip is normal if it is close to *any* normal condition). The final score is
weight_arc * mean(CNN branch scores) + weight_cls * classical score (normalised by the weights). Higher = more anomalous.

Calibration: ``p_value`` compares a score with scores of held-out normal clips (split conformal): it is the
fraction of normal recordings that look at least this unusual, with a finite-sample guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def is_cosine(branch: str) -> bool:
    return branch.startswith("arc")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _dist(branch: str, queries: np.ndarray, bank: np.ndarray) -> np.ndarray:
    if is_cosine(branch):  # both sides are L2-normalised
        return 1.0 - queries @ bank.T
    sq = (queries**2).sum(1)[:, None] + (bank**2).sum(1)[None, :] - 2 * queries @ bank.T
    return np.sqrt(np.maximum(sq, 0.0))


def leave_one_out_stats(branch: str, bank: np.ndarray) -> tuple[float, float]:
    """Mean/std of each normal clip's distance to its nearest *other* normal clip."""
    if len(bank) < 3:
        return 0.0, 1.0
    d = _dist(branch, bank, bank)
    np.fill_diagonal(d, np.inf)
    loo = d.min(axis=1)
    return float(loo.mean()), float(loo.std() + 1e-9)


class EnsembleScorer:
    def __init__(self, banks: dict[str, np.ndarray], stats: dict[str, Any], weights: dict[str, float]):
        """banks: keys '<branch>__<group>__<domain>'; stats: {'<branch>': {'<group>': [mu, sd]}}."""
        self.weights = weights
        self.stats = stats
        self.raw_banks = {k: np.asarray(v, dtype=np.float32) for k, v in banks.items()}
        self.banks: dict[str, list[tuple[str, np.ndarray]]] = {}
        for key, arr in self.raw_banks.items():
            branch, group, _domain = key.split("__", 2)
            self.banks.setdefault(branch, []).append((group, arr))
        self.cnn_branches = sorted(b for b in self.banks if is_cosine(b))
        self.conditions = sorted({tuple(k.split("__", 2)[1:]) for k in self.raw_banks})

    def with_extra(self, banks: dict[str, np.ndarray], stats: dict[str, dict[str, list[float]]]) -> EnsembleScorer:
        """A new scorer that also knows extra (e.g. enrolled) conditions."""
        merged_stats = {b: {**self.stats.get(b, {}), **stats.get(b, {})} for b in set(self.stats) | set(stats)}
        return EnsembleScorer({**self.raw_banks, **banks}, merged_stats, self.weights)

    def nearest_condition(self, views: dict[str, np.ndarray]) -> list[str]:
        """'<group>__<domain>' of the bank each query is closest to (combined z-score over branches)."""
        per = []
        for group, domain in self.conditions:
            def z(b, g=group, dm=domain):
                mu, sd = self.stats[b][g]
                return (_dist(b, views[b], self.raw_banks[f"{b}__{g}__{dm}"]).min(axis=1) - mu) / sd

            cnn = np.mean([z(b) for b in self.cnn_branches], axis=0)
            per.append((self.weights["arc"] * cnn + self.weights["cls"] * z("cls")) / (self.weights["arc"] + self.weights["cls"]))
        best = np.argmin(np.stack(per, axis=1), axis=1)
        return [f"{self.conditions[i][0]}__{self.conditions[i][1]}" for i in best]

    @classmethod
    def load(cls, bank_path: str | Path, stats: dict[str, Any], weights: dict[str, float]) -> EnsembleScorer:
        with np.load(bank_path) as z:
            return cls({k: z[k] for k in z.files}, stats, weights)

    def branch_scores(self, branch: str, queries: np.ndarray) -> np.ndarray:
        best = np.full(len(queries), np.inf)
        for group, bank in self.banks[branch]:
            mu, sd = self.stats[branch][group]
            z = (_dist(branch, queries, bank).min(axis=1) - mu) / sd
            best = np.minimum(best, z)
        return best

    def score(self, views: dict[str, np.ndarray]) -> np.ndarray:
        """views: {'arc0': emb, 'arc1': emb, ..., 'cls': standardised mean spectrum} -> (N,) scores."""
        cnn = np.mean([self.branch_scores(b, views[b]) for b in self.cnn_branches], axis=0)
        w_arc, w_cls = self.weights["arc"], self.weights["cls"]
        return (w_arc * cnn + w_cls * self.branch_scores("cls", views["cls"])) / (w_arc + w_cls)


def conformal_p_value(score: float, calibration_scores: Sequence[float]) -> float:
    """P(normal clip scores >= score), finite-sample valid: (1 + #{cal >= s}) / (n + 1)."""
    cal = np.asarray(calibration_scores)
    return float((1 + np.sum(cal >= score)) / (len(cal) + 1))


def severity_from_p(p: float, alpha: float) -> float:
    """1.0 exactly at the alert level (p == alpha), 2.0 at p == alpha^2, 0 for p == 1. Monotone in p."""
    return float(np.log(max(p, 1e-12)) / np.log(alpha)) if p < 1 else 0.0
