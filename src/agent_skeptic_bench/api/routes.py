"""API routes for Agent Skeptic Bench."""

from fastapi import FastAPI

from .agents import router as agents_router
from .evaluations import router as evaluations_router
from .metrics import router as metrics_router
from .scenarios import router as scenarios_router
from .sessions import router as sessions_router


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    # API v1 routes
    api_prefix = "/api/v1"

    app.include_router(
        scenarios_router,
        prefix=f"{api_prefix}/scenarios",
        tags=["Scenarios"]
    )

    app.include_router(
        evaluations_router,
        prefix=f"{api_prefix}/evaluations",
        tags=["Evaluations"]
    )

    app.include_router(
        sessions_router,
        prefix=f"{api_prefix}/sessions",
        tags=["Sessions"]
    )

    app.include_router(
        agents_router,
        prefix=f"{api_prefix}/agents",
        tags=["Agents"]
    )

    app.include_router(
        metrics_router,
        prefix=f"{api_prefix}/metrics",
        tags=["Metrics"]
    )
