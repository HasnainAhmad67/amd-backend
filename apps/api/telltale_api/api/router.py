from fastapi import APIRouter

from .routes import acoustic, ai, alerts, assessments, health, machines, models, sensors

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(sensors.router, prefix="/sensors", tags=["sensors"])
api_router.include_router(acoustic.router, prefix="/acoustic", tags=["acoustic"])
api_router.include_router(machines.router, prefix="/machines", tags=["fleet"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
api_router.include_router(assessments.router, prefix="/assessments", tags=["assessments"])
api_router.include_router(ai.router, prefix="/ai", tags=["ai"])
