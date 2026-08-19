# src/nw_rag/api/routes.py
#
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nw_rag.core.ingestion import DocumentIngester
from nw_rag.services.rag_engine import RAGEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["query"])

ingester = DocumentIngester()
if not ingester.load_index():
    logger.info("Building index...")
    count = ingester.ingest()
    ingester.save_index()
    logger.info(f"Ingested {count} chunks")

rag_engine = RAGEngine(ingester)

class QueryRequest(BaseModel):
    question: str
    stream: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    timing: dict
    tokens: dict
    fallback: bool

@router.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"Query: {request.question[:100]}...")
        result = rag_engine.query(request.question)

        logger.info(
            f"Confidence: {result['confidence']:.3f} | "
            f"Fallback: {result['fallback']} | "
            f"Retrieval: {result['timing']['retrieval']:.3f}s | "
            f"Generation: {result['timing']['generation']:.3f}s"
        )

        if request.stream:
            return StreamingResponse(
                stream_response(result),
                media_type="text/event-stream"
            )

        return result

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def stream_response(result: dict) -> AsyncGenerator[str, None]:
    answer = result["answer"]
    for char in answer:
        yield f"data: {json.dumps({'chunk': char})}\n\n"

    metadata = {
        "sources": result["sources"],
        "confidence": result["confidence"],
        "timing": result["timing"],
        "fallback": result["fallback"]
    }
    yield f"data: {json.dumps({'done': True, 'metadata': metadata})}\n\n"
