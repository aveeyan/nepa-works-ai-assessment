# src/nw_rag/services/rag_engine.py

# Standard Imports
import re
import time
from typing import Dict, Any, List, Tuple
import uuid

# Third-Party Imports
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Local Imports
from nw_rag.core.ingestion import DocumentIngester
from nw_rag.config import Config
from nw_rag.services.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    FALLBACK_INSUFFICIENT_CONTEXT,
    FALLBACK_LOW_CONFIDENCE,
    FALLBACK_NO_DOCUMENTS,
    FALLBACK_INJECTION_DETECTED,
    INJECTION_PATTERNS
)
from nw_rag.logging_config import StructuredLogger, logger

class RAGEngine:
    """Handles the RAG logic: retrieval, LLM generation, and fallback guardrails."""

    def __init__(self, ingester: DocumentIngester):
        self.ingester = ingester
        self.llm = ChatOllama(
            model=Config.LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.1
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_PROMPT)
        ])
        self.request_id = None

    def _detect_prompt_injection(self, query: str) -> bool:
        """Detects potential prompt injection attempts."""
        query_lower = query.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, query_lower):
                StructuredLogger.log_prompt_injection_detected(query)
                return True
        return False

    def _format_context(self, docs: List[Tuple[Document, float]]) -> str:
        """Formats retrieved documents into a context string."""
        formatted = []
        for doc, score in docs:
            source = doc.metadata.get('source', 'unknown')
            content = doc.page_content
            formatted.append(f"[Source: {source} | Score: {score:.3f}]\n{content}")
        return "\n\n".join(formatted)

    def _extract_token_usage(self, response) -> Dict[str, int]:
        """Extracts token usage from Ollama response metadata."""
        tokens = {"prompt": 0, "completion": 0}
        try:
            metadata = response.response_metadata if hasattr(response, 'response_metadata') else {}

            if 'token_usage' in metadata:
                usage = metadata['token_usage']
                tokens['prompt'] = usage.get('prompt_tokens', 0)
                tokens['completion'] = usage.get('completion_tokens', 0)
            elif 'eval_count' in metadata:
                tokens['prompt'] = metadata.get('prompt_eval_count', 0)
                tokens['completion'] = metadata.get('eval_count', 0)
        except Exception as e:
            logger.debug(f"Could not extract token usage: {e}")

        return tokens

    def _build_response(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        confidence: float,
        retrieval_time: float,
        generation_time: float,
        total_time: float,
        tokens: Dict[str, int] = None,
        fallback: bool = False
    ) -> Dict[str, Any]:
        """Builds a standardized response dictionary."""
        if tokens is None:
            tokens = {"prompt": 0, "completion": 0}

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "timing": {
                "retrieval": retrieval_time,
                "generation": generation_time,
                "total": total_time
            },
            "tokens": tokens,
            "fallback": fallback
        }

    def query(self, question: str) -> Dict[str, Any]:
        """Processes a query through the RAG pipeline."""
        self.request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.debug(f"Processing query | RequestID: {self.request_id}")
        StructuredLogger.log_request(question, stream=False, request_id=self.request_id)

        # Safety check
        if self._detect_prompt_injection(question):
            return self._build_response(
                answer=FALLBACK_INJECTION_DETECTED,
                sources=[],
                confidence=0.0,
                retrieval_time=0,
                generation_time=0,
                total_time=time.time() - start_time,
                fallback=True
            )

        # Retrieval
        retrieval_start = time.time()
        results = self.ingester.retrieve(question)
        retrieval_time = time.time() - retrieval_start

        StructuredLogger.log_retrieval(question, results, retrieval_time)

        # Check for no results
        if not results:
            StructuredLogger.log_fallback("No documents found", 0.0)
            return self._build_response(
                answer=FALLBACK_NO_DOCUMENTS,
                sources=[],
                confidence=0.0,
                retrieval_time=retrieval_time,
                generation_time=0,
                total_time=time.time() - start_time,
                fallback=True
            )

        # Check similarity threshold
        top_score = results[0][1]
        if top_score < Config.SIMILARITY_THRESHOLD:
            sources = [
                {"source": doc.metadata.get('source', 'unknown'), "score": float(score)}
                for doc, score in results[:3]
            ]
            StructuredLogger.log_fallback(
                "Low confidence score",
                top_score,
                Config.SIMILARITY_THRESHOLD
            )
            return self._build_response(
                answer=FALLBACK_LOW_CONFIDENCE.format(
                    score=top_score,
                    threshold=Config.SIMILARITY_THRESHOLD
                ),
                sources=sources,
                confidence=float(top_score),
                retrieval_time=retrieval_time,
                generation_time=0,
                total_time=time.time() - start_time,
                fallback=True
            )

        # Generate response
        context = self._format_context(results[:Config.TOP_K])
        generation_start = time.time()

        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })
        generation_time = time.time() - generation_start

        # Extract token usage
        tokens = self._extract_token_usage(response)
        StructuredLogger.log_generation(question, response.content, generation_time, tokens)

        # Prepare sources
        sources = [
            {
                "source": doc.metadata.get('source', 'unknown'),
                "score": float(score)
            }
            for doc, score in results[:Config.TOP_K]
        ]

        result = self._build_response(
            answer=response.content,
            sources=sources,
            confidence=float(top_score),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=time.time() - start_time,
            tokens=tokens,
            fallback=False
        )

        StructuredLogger.log_query_complete(result, time.time() - start_time)

        return result
