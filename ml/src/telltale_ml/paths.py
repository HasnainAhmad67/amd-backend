"""Filesystem layout. Every path the ML code touches is resolved here.

Override with environment variables (Docker does this):
    TELLTALE_DATA_DIR    default: <repo>/data
    TELLTALE_MODELS_DIR  default: <repo>/models
    TELLTALE_CONFIG_DIR  default: <repo>/ml/configs
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "AGENTS.md").is_file() and (parent / "ml").is_dir():
            return parent
    # Non-editable install (e.g. inside the API image): fall back to the working directory.
    return Path.cwd()


REPO_ROOT = _find_repo_root()


def data_dir() -> Path:
    return Path(os.environ.get("TELLTALE_DATA_DIR", REPO_ROOT / "data"))


def models_dir() -> Path:
    return Path(os.environ.get("TELLTALE_MODELS_DIR", REPO_ROOT / "models"))


def raw_dir() -> Path:
    return data_dir() / "raw"


def processed_dir() -> Path:
    return data_dir() / "processed"


def config_dir() -> Path:
    return Path(os.environ.get("TELLTALE_CONFIG_DIR", REPO_ROOT / "ml" / "configs"))


def default_config(name: str) -> Path:
    """Path of a bundled YAML config, e.g. ``default_config("sensor")``."""
    path = config_dir() / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config {path} not found. Set TELLTALE_CONFIG_DIR to the folder with the ml/configs files.")
    return path
