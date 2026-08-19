# src/nw_rag/main.py

import logging
from fastapi import FastAPI
from nw_rag.api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aperture Science RAG API",
    description="RAG API for Aperture Science documentation",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
async def root():
    return {
        "message": "Aperture Science RAG API",
        "endpoints": {
            "POST /api/v1/query": "Query the knowledge base"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
