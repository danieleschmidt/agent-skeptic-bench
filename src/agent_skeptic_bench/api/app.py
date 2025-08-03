"""FastAPI application factory for Agent Skeptic Bench."""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..database.connection import get_database
from ..cache import initialize_cache, close_cache
from .middleware import setup_middleware
from .routes import register_routes


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Agent Skeptic Bench API...")
    
    try:
        # Initialize database
        db = get_database()
        await db.create_tables()
        logger.info("Database initialized")
        
        # Initialize cache
        await initialize_cache()
        logger.info("Cache initialized")
        
        # Health check
        db_healthy = await db.health_check()
        if not db_healthy:
            logger.warning("Database health check failed")
        
        logger.info("API startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agent Skeptic Bench API...")
    
    try:
        # Close cache connections
        await close_cache()
        logger.info("Cache connections closed")
        
        # Close database connections
        db = get_database()
        await db.close()
        logger.info("Database connections closed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("API shutdown complete")


def create_app(
    title: str = "Agent Skeptic Bench API",
    description: str = "API for evaluating AI agents' epistemic vigilance and skepticism",
    version: str = "1.0.0",
    debug: bool = False
) -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        debug=debug,
        lifespan=lifespan,
        docs_url="/docs" if debug else None,
        redoc_url="/redoc" if debug else None,
        openapi_url="/openapi.json" if debug else None
    )
    
    # CORS middleware
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup custom middleware
    setup_middleware(app)
    
    # Register routes
    register_routes(app)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "type": "internal_error"
            }
        )
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        try:
            # Check database
            db = get_database()
            db_healthy = await db.health_check()
            
            # Check cache
            from ..cache import get_cache_manager
            cache_manager = get_cache_manager()
            cache_status = await cache_manager.health_check()
            
            status = "healthy" if db_healthy else "unhealthy"
            
            return {
                "status": status,
                "timestamp": "2025-01-01T00:00:00Z",  # Will be actual timestamp
                "version": version,
                "database": {
                    "status": "healthy" if db_healthy else "unhealthy"
                },
                "cache": cache_status
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e)
                }
            )
    
    logger.info(f"FastAPI application created: {title} v{version}")
    return app


def create_production_app() -> FastAPI:
    """Create production-ready FastAPI application."""
    return create_app(
        debug=False,
        title="Agent Skeptic Bench API",
        description="Production API for AI agent skepticism evaluation"
    )


def create_development_app() -> FastAPI:
    """Create development FastAPI application.""" 
    return create_app(
        debug=True,
        title="Agent Skeptic Bench API (Development)",
        description="Development API for AI agent skepticism evaluation"
    )