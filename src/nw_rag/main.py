# src/nw_rag/main.py

# Standard Imports
from contextlib import asynccontextmanager
from pathlib import Path

# Third-Party Imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local Imports
from nw_rag.api.routes import router
from nw_rag.logging_config import StructuredLogger, logger

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    StructuredLogger.log_system_event("Application Starting", {
        "version": "1.0.0",
        "environment": "development"
    })
    logger.info("=" * 60)
    logger.info("Aperture Science RAG API Starting...")
    logger.info("=" * 60)

    yield

    # Shutdown
    StructuredLogger.log_system_event("Application Shutting Down", {})
    logger.info("Aperture Science RAG API Shutting Down...")

# Application
app = FastAPI(
    title="Aperture Science RAG API",
    description="RAG API for Aperture Science documentation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with dynamic endpoint listing."""
    endpoints = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            if route.path in ["/", "/health", "/logs", "/docs", "/redoc", "/openapi.json"]:
                continue
            methods = list(route.methods) if route.methods else []
            endpoints.append({
                "path": route.path,
                "methods": methods,
                "name": route.name if hasattr(route, "name") else "Unknown"
            })

    return {
        "message": "Aperture Science RAG API",
        "version": "1.0.0",
        "endpoints": endpoints
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "llama3.2:3b"
    }


@app.get("/logs")
async def get_logs():
    """Get recent logs (for debugging - should be secured in production)."""
    log_files = []
    log_dir = Path("logs")
    if log_dir.exists():
        for file in log_dir.glob("*.json"):
            log_files.append({
                "name": file.name,
                "size": file.stat().st_size,
                "modified": file.stat().st_mtime
            })
        for file in log_dir.glob("*.log"):
            log_files.append({
                "name": file.name,
                "size": file.stat().st_size,
                "modified": file.stat().st_mtime
            })
    return {
        "log_files": sorted(log_files, key=lambda x: x["modified"], reverse=True),
        "log_dir": str(log_dir.absolute())
    }
