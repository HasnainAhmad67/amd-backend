"""Discover labelled clips in MIMII / MIMII DUE folder trees.

Both loaders scan recursively and parse labels from paths, so they tolerate an extra top-level
folder from unzipping (e.g. ``dev_data/fan/...``).

MIMII DUE (Zenodo 4740355, DCASE 2021 T2), mono 16 kHz, 10 s:
    <root>/fan/train/section_00_source_train_normal_0000_<attrs>.wav
    <root>/fan/source_test/section_00_source_test_anomaly_0003.wav
    <root>/fan/target_test/section_02_target_test_normal_0011.wav

MIMII (Zenodo 3384388), 8-channel 16 kHz, 10 s:
    <root>/<snr>/fan/id_00/normal/00000000.wav
    <root>/<snr>/fan/id_00/abnormal/00000000.wav
    (no official split -> we hold out as many normals per machine id as there are abnormals)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SECTION_RE = re.compile(r"section_(\d+)_(source|target)_(train|test)_(normal|anomaly)")


@dataclass(frozen=True)
class Clip:
    path: Path
    label: str  # "normal" | "anomaly"
    split: str  # "train" | "test"
    group: str  # machine id / section, used for per-group metrics
    domain: str = "source"  # MIMII DUE source/target domain; "source" for MIMII


def discover_mimii_due(root: Path, machine_type: str) -> list[Clip]:
    clips = []
    for wav in sorted(root.rglob("*.wav")):
        if machine_type not in wav.parts:
            continue
        m = SECTION_RE.search(wav.name)
        if not m:
            continue
        section, domain, split, label = m.groups()
        clips.append(Clip(wav, label, split, f"section_{section}", domain))
    return clips


def discover_mimii(root: Path, machine_type: str, seed: int = 0) -> list[Clip]:
    by_id: dict[str, dict[str, list[Path]]] = defaultdict(lambda: {"normal": [], "abnormal": []})
    for wav in sorted(root.rglob("*.wav")):
        parts = wav.parts
        if machine_type not in parts or wav.parent.name not in ("normal", "abnormal"):
            continue
        machine_id = next((p for p in parts if p.startswith("id_")), "id_unknown")
        by_id[machine_id][wav.parent.name].append(wav)

    rng = np.random.default_rng(seed)
    clips: list[Clip] = []
    for machine_id, files in sorted(by_id.items()):
        normals = list(files["normal"])
        rng.shuffle(normals)
        n_test = len(files["abnormal"])
        clips += [Clip(p, "normal", "test", machine_id) for p in normals[:n_test]]
        clips += [Clip(p, "normal", "train", machine_id) for p in normals[n_test:]]
        clips += [Clip(p, "anomaly", "test", machine_id) for p in files["abnormal"]]
    return clips


def discover(dataset: str, root: Path, machine_type: str, seed: int = 0) -> list[Clip]:
    if dataset == "mimii_due":
        clips = discover_mimii_due(root, machine_type)
    elif dataset == "mimii":
        clips = discover_mimii(root, machine_type, seed)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}; expected 'mimii_due' or 'mimii'")
    if not clips:
        raise FileNotFoundError(
            f"No {dataset} clips for machine type {machine_type!r} under {root}. "
            f"Run `telltale-download {dataset.replace('_', '-')} --machine {machine_type}` first."
        )
    return clips
