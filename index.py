"""Vercel Python Function entrypoint for Telltale's FastAPI backend."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# The API and ML packages stay in their project folders; make their source modules
# importable from Vercel's root-level Python function.
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "ml" / "src"))

# Public demo defaults. Override in Vercel project settings if needed.
os.environ.setdefault("TELLTALE_MODELS_DIR", str(ROOT / "models"))
os.environ.setdefault("TELLTALE_DATA_DIR", "/tmp/telltale-data")
os.environ.setdefault("TELLTALE_DEMO_MODE", "true")
os.environ.setdefault("TELLTALE_SENSOR_AUTOTRAIN_OFFLINE", "true")
os.environ.setdefault("TELLTALE_CORS_ORIGINS", '["*"]')
os.environ.setdefault("TELLTALE_AI_PROVIDER", "template")

from telltale_api.main import app  # noqa: E402
