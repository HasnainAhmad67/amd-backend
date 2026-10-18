"""Vercel Python Function entrypoint for Telltale's FastAPI backend."""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# LightGBM ko libgomp chahiye. Vercel par app import hone se pehle ise load karein.
_libgomp = ROOT / "vendor" / "lib" / "libgomp.so.1"
if _libgomp.is_file():
    ctypes.CDLL(str(_libgomp), mode=ctypes.RTLD_GLOBAL)

# API aur ML packages ke source folders ko importable banayein.
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "ml" / "src"))

# Public demo defaults. Vercel environment variables se override ho sakte hain.
os.environ.setdefault("TELLTALE_MODELS_DIR", str(ROOT / "models"))
os.environ.setdefault("TELLTALE_DATA_DIR", "/tmp/telltale-data")
os.environ.setdefault("TELLTALE_DEMO_MODE", "true")
os.environ.setdefault("TELLTALE_SENSOR_AUTOTRAIN_OFFLINE", "true")
os.environ.setdefault("TELLTALE_CORS_ORIGINS", '["*"]')
os.environ.setdefault("TELLTALE_AI_PROVIDER", "template")

from telltale_api.main import app  # noqa: E402
