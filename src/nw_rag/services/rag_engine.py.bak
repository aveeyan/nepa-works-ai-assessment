# src/nw_rag/services/rag_engine.py

# Standard Imports
import re
import time
from typing import Dict, Any, List, Tuple

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
    FALLBACK_LOW_CONFIDENCE,
    FALLBACK_NO_DOCUMENTS,
    FALLBACK_INJECTION_DETECTED,
    INJECTION_PATTERNS
)

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

    def _detect_prompt_injection(self, query: str) -> bool:
        """Detects potential prompt injection attempts."""
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in INJECTION_PATTERNS)

    def _format_context(self, docs: List[Tuple[Document, float]]) -> str:
        """Formats retrieved documents into a context string."""
        formatted = []
        for doc, score in docs:
            source = doc.metadata.get('source', 'unknown')
            content = doc.page_content
            formatted.append(f"[Source: {source} | Score: {score:.3f}]\n{content}")
        return "\n\n".join(formatted)

    def _build_response(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        confidence: float,
        retrieval_time: float,
        generation_time: float,
        total_time: float,
        fallback: bool = False
    ) -> Dict[str, Any]:
        """Builds a standardized response dictionary."""
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "timing": {
                "retrieval": retrieval_time,
                "generation": generation_time,
                "total": total_time
            },
            "tokens": {
                "prompt": 0,
                "completion": 0
            },
            "fallback": fallback
        }

    def query(self, question: str) -> Dict[str, Any]:
        """Processes a query through the RAG pipeline."""
        start_time = time.time()

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

        # Check for no results
        if not results:
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

        # Prepare sources
        sources = [
            {
                "source": doc.metadata.get('source', 'unknown'),
                "score": float(score)
            }
            for doc, score in results[:Config.TOP_K]
        ]

        return self._build_response(
            answer=response.content,
            sources=sources,
            confidence=float(top_score),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=time.time() - start_time,
            fallback=False
        )
