"""FastAPI entrypoint:  uvicorn telltale_api.main:app --reload"""

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api.deps import AppState
from .api.router import api_router
from .core.config import Settings, get_settings
from .services.ai_output_store import AiOutputStore
from .services.alerts import AlertStore, SqliteAlertStore
from .services.assessment_store import AssessmentStore
from .services.bootstrap import ensure_sensor_model
from .services.fleet import FleetSimulator, load_specs
from .services.narrator import Narrator
from .services.playbook import Playbook
from .services.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def build_state(settings: Settings) -> AppState:
    registry = ModelRegistry(settings.models_dir, fallback_dir=settings.fallback_models_dir).load()
    alerts = AlertStore()
    state_db = settings.state_db or settings.data_dir / "state" / "telltale.db"
    if settings.persist_alerts:
        try:
            alerts = SqliteAlertStore(state_db)
        except Exception:  # read-only disk, locked file, ...: keep the app working without persistence
            logging.getLogger("telltale").exception("Could not open %s; alerts will not survive restarts", state_db)
    fleet = FleetSimulator(load_specs(settings.fleet_file), registry, alerts, seed=settings.simulator_seed)
    # Assessments share the same SQLite file (schema is versioned separately).
    assessments: AssessmentStore | None = None
    try:
        assessments = AssessmentStore(state_db)
    except Exception:  # read-only disk, locked file, ...: routes report 503 instead
        logging.getLogger("telltale").exception("Could not open %s; assessments will not persist", state_db)
    # AI outputs append to the same SQLite file as a NEW versioned table (v3);
    # assessment tables/records are never altered and no in-place migration runs.
    ai_outputs: AiOutputStore | None = None
    try:
        ai_outputs = AiOutputStore(state_db)
    except Exception:  # read-only disk, locked file, ...: routes report 503 instead
        logging.getLogger("telltale").exception("Could not open %s; AI outputs will not persist", state_db)
    return AppState(
        settings=settings,
        registry=registry,
        playbook=Playbook(settings.playbook_file),
        alerts=alerts,
        fleet=fleet,
        narrator=Narrator(settings),
        assessments=assessments,
        ai_outputs=ai_outputs,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = build_state(settings)
        app.state.telltale = state
        tasks = [asyncio.create_task(ensure_sensor_model(settings, state.registry, state.bootstrap))]
        if settings.simulator_enabled:
            state.fleet.tick()  # populate immediately so the first page load has data
            tasks.append(asyncio.create_task(state.fleet.run(settings.simulator_interval_s)))
        yield
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(
        title="Telltale API",
        version=__version__,
        description="Predictive maintenance from machine telemetry and machine sound.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
