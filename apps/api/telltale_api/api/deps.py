"""Shared application state, built once at startup and injected into routes."""

from dataclasses import dataclass, field

from fastapi import Request

from ..core.config import Settings
from ..services.ai_output_store import AiOutputStore
from ..services.alerts import AlertStore
from ..services.assessment_store import AssessmentStore
from ..services.bootstrap import BootstrapStatus
from ..services.fleet import FleetSimulator
from ..services.narrator import Narrator
from ..services.playbook import Playbook
from ..services.registry import ModelRegistry


@dataclass
class AppState:
    settings: Settings
    registry: ModelRegistry
    playbook: Playbook
    alerts: AlertStore
    fleet: FleetSimulator
    narrator: Narrator
    bootstrap: BootstrapStatus = field(default_factory=BootstrapStatus)
    assessments: AssessmentStore | None = None  # None if the DB could not be opened
    ai_outputs: AiOutputStore | None = None  # None if the DB could not be opened


def get_state(request: Request) -> AppState:
    return request.app.state.telltale
