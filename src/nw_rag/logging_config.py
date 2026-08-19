# src/nw_rag/logging_config.py

# Standard Imports
import sys
import json
from datetime import datetime
from pathlib import Path

# Third-Party Imports
from loguru import logger
logger.remove()

# Local Imports
from .config import Config


LOG_FORMAT = Config.LOG_FORMAT

# Console handler - Human readable
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level="INFO",
    colorize=True
)

# File handler - Structured JSON for analytics
logger.add(
    "logs/rag_api_{time:YYYY-MM-DD}.json",
    format="{time} | {level} | {name}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="1 day",
    retention="30 days",
    serialize=True  # JSON format for structured logging
)

# File handler - Detailed debug
logger.add(
    "logs/debug_{time:YYYY-MM-DD}.log",
    format=LOG_FORMAT,
    level="DEBUG",
    rotation="1 day",
    retention="7 days"
)

# File handler - Error only
logger.add(
    "logs/errors_{time:YYYY-MM-DD}.log",
    format=LOG_FORMAT,
    level="ERROR",
    rotation="1 day",
    retention="30 days"
)


class StructuredLogger:
    """Wrapper for structured logging with context."""

    @staticmethod
    def log_request(question: str, stream: bool, request_id: str = None):
        """Log incoming API request."""
        logger.info(
            f"API Request | RequestID: {request_id or 'N/A'} | "
            f"Question: {question[:100]}{'...' if len(question) > 100 else ''} | "
            f"Stream: {stream}"
        )

    @staticmethod
    def log_retrieval(query: str, results: list, retrieval_time: float):
        """Log retrieval phase metrics."""
        logger.info(
            f"Retrieval | Query: {query[:50]}... | "
            f"Results: {len(results)} | "
            f"Time: {retrieval_time:.3f}s"
        )

        # Log detailed retrieval results at debug level
        for idx, (doc, score) in enumerate(results[:3]):
            logger.debug(
                f"Retrieval Result {idx+1} | "
                f"Source: {doc.metadata.get('source', 'unknown')} | "
                f"Score: {score:.3f} | "
                f"Content: {doc.page_content[:100]}..."
            )

    @staticmethod
    def log_generation(question: str, response: str, generation_time: float, tokens: dict):
        """Log generation phase metrics."""
        logger.info(
            f"Generation | Query: {question[:50]}... | "
            f"Response Length: {len(response)} chars | "
            f"Time: {generation_time:.3f}s | "
            f"Prompt Tokens: {tokens.get('prompt', 0)} | "
            f"Completion Tokens: {tokens.get('completion', 0)}"
        )

    @staticmethod
    def log_fallback(reason: str, confidence: float, threshold: float = None):
        """Log fallback triggers."""
        logger.warning(
            f"Fallback Triggered | Reason: {reason} | "
            f"Confidence: {confidence:.3f} | "
            f"Threshold: {threshold or 'N/A'}"
        )

    @staticmethod
    def log_query_complete(result: dict, total_time: float):
        """Log complete query metrics."""
        logger.info(
            f"Query Complete | "
            f"Confidence: {result['confidence']:.3f} | "
            f"Fallback: {result['fallback']} | "
            f"Sources: {len(result['sources'])} | "
            f"Total Time: {total_time:.3f}s | "
            f"Retrieval: {result['timing']['retrieval']:.3f}s | "
            f"Generation: {result['timing']['generation']:.3f}s"
        )

    @staticmethod
    def log_prompt_injection_detected(question: str):
        """Log prompt injection attempts."""
        logger.warning(
            f"Prompt Injection Detected | "
            f"Question: {question[:100]}{'...' if len(question) > 100 else ''}"
        )

    @staticmethod
    def log_error(error: Exception, context: dict = None):
        """Log errors with context."""
        logger.error(
            f"Error: {str(error)} | "
            f"Context: {json.dumps(context or {})}"
        )

    @staticmethod
    def log_system_event(event: str, details: dict = None):
        """Log system events (startup, shutdown, etc.)."""
        logger.info(
            f"System Event: {event} | "
            f"Details: {json.dumps(details or {})}"
        )

# Create logs directory if not existing
Path("logs").mkdir(exist_ok=True)

# Export logger instance
__all__ = ["logger", "StructuredLogger"]
