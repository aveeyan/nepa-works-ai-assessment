# src/nw_rag/api/routes.py

# Standard Imports
import json
import logging
from typing import AsyncGenerator
import uuid

# Third-Party Imports
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Local Imports
from nw_rag.core.ingestion import DocumentIngester
from nw_rag.services.rag_engine import RAGEngine
from nw_rag.logging_config import StructuredLogger, logger

# Router
router = APIRouter(prefix="/api/v1", tags=["query"])

# Initialize components
ingester = DocumentIngester()
if not ingester.load_index():
    logger.info("Building index...")
    count = ingester.ingest()
    ingester.save_index()
    logger.info(f"Ingested {count} chunks")

rag_engine = RAGEngine(ingester)

# Pydantic Models
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
async def query_endpoint(request: Request, query_req: QueryRequest):
    """Query the RAG system."""
    request_id = str(uuid.uuid4())[:8]

    try:
        StructuredLogger.log_request(
            query_req.question,
            query_req.stream,
            request_id
        )

        logger.debug(f"RequestID: {request_id} | Client: {request.client.host}")

        result = rag_engine.query(query_req.question)

        StructuredLogger.log_query_complete(result, result['timing']['total'])

        if query_req.stream:
            return StreamingResponse(
                stream_response(result, request_id),
                media_type="text/event-stream"
            )

        return result

    except Exception as e:
        StructuredLogger.log_error(e, {"request_id": request_id, "question": query_req.question})
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(result: dict, request_id: str) -> AsyncGenerator[str, None]:
    """Stream the response as Server-Sent Events."""
    try:
        answer = result["answer"]
        for char in answer:
            yield f"data: {json.dumps({'chunk': char})}\n\n"

        metadata = {
            "sources": result["sources"],
            "confidence": result["confidence"],
            "timing": result["timing"],
            "fallback": result["fallback"],
            "request_id": request_id
        }
        yield f"data: {json.dumps({'done': True, 'metadata': metadata})}\n\n"

        StructuredLogger.log_generation(
            "streaming",
            answer,
            result['timing']['generation'],
            result['tokens']
        )

    except Exception as e:
        StructuredLogger.log_error(e, {"request_id": request_id})
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
