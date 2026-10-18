"""First-start helper: make sure a sensor model exists so the dashboard works right away.

Runs in a worker thread after startup. While it runs, /health reports
``sensor_bootstrap.status == "running"`` and the UI shows a "getting ready" banner.
"""

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi.concurrency import run_in_threadpool

from ..core.config import Settings
from .registry import ModelRegistry

log = logging.getLogger("telltale.bootstrap")


@dataclass
class BootstrapStatus:
    status: str = "idle"  # idle | running | done | failed
    message: str = ""


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".write-test-{uuid.uuid4().hex}"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


async def ensure_sensor_model(settings: Settings, registry: ModelRegistry, status: BootstrapStatus) -> None:
    if registry.sensor is not None or not settings.sensor_autotrain:
        return
    out_dir = settings.models_dir / "sensor"
    if not _writable(out_dir):
        out_dir = settings.fallback_models_dir / "sensor"
        log.warning("%s is not writable; saving the auto-trained sensor model to %s", settings.models_dir, out_dir)
    status.status, status.message = "running", "Training the sensor risk model (first start only, about a minute)..."
    log.info(status.message)
    try:
        # Keep training-only dependencies (notably scikit-learn) out of API startup.
        from telltale_ml.sensor.train import train_default

        meta = await run_in_threadpool(train_default, out_dir, settings.sensor_autotrain_offline)
        registry.load()
        if registry.sensor is None:
            raise RuntimeError(registry.errors.get("sensor", "model trained but could not be loaded"))
        status.status = "done"
        status.message = f"Sensor model ready (trained on {meta['data_source']})."
        log.info(status.message)
    except Exception as exc:
        log.exception("Automatic sensor-model training failed")
        status.status, status.message = "failed", f"Automatic training failed: {type(exc).__name__}: {exc}"
