# src/nw_rag/core/retrieval.py

# Standard Imports
from typing import List, Tuple, Optional

# Third-Party Imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Local Imports
from nw_rag.config import Config
from nw_rag.logging_config import logger


class VectorRetriever:
    """Handles vector retrieval operations."""

    def __init__(self, vectorstore: Optional[FAISS] = None):
        self.vectorstore = vectorstore
        self.embeddings = OllamaEmbeddings(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.EMBEDDING_MODEL
        )

    def set_vectorstore(self, vectorstore: FAISS):
        """Set the vectorstore instance."""
        self.vectorstore = vectorstore

    def retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k relevant documents for a query.

        Args:
            query: The search query
            k: Number of documents to retrieve (default: from Config)

        Returns:
            List of (Document, score) tuples
        """
        k = k or Config.TOP_K

        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Load or build first.")

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            raise

    def retrieve_with_threshold(
        self,
        query: str,
        k: int = None,
        threshold: float = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with a similarity threshold filter.

        Args:
            query: The search query
            k: Number of documents to retrieve
            threshold: Minimum similarity score (default: from Config)

        Returns:
            List of (Document, score) tuples filtered by threshold
        """
        k = k or Config.TOP_K
        threshold = threshold or Config.SIMILARITY_THRESHOLD

        results = self.retrieve(query, k)

        # Filter by threshold
        filtered = [(doc, score) for doc, score in results if score >= threshold]

        logger.debug(f"Filtered {len(filtered)} documents (threshold: {threshold})")
        return filtered

    def get_top_score(self, query: str, k: int = None) -> float:
        """
        Get the highest similarity score for a query.

        Args:
            query: The search query
            k: Number of documents to retrieve

        Returns:
            Highest similarity score
        """
        results = self.retrieve(query, k or 1)
        if results:
            return results[0][1]
        return 0.0

    def load_index(self, path: str = None) -> bool:
        """Load vectorstore index from disk."""
        path = path or Config.FAISS_INDEX_PATH

        try:
            self.vectorstore = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=Config.ALLOW_DANGEROUS_DESERIALIZATION
            )
            logger.info(f"Loaded index from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index from {path}: {e}")
            return False

    def save_index(self, path: str = None) -> bool:
        """Save vectorstore index to disk."""
        path = path or Config.FAISS_INDEX_PATH

        if not self.vectorstore:
            logger.warning("No vectorstore to save")
            return False

        try:
            self.vectorstore.save_local(path)
            logger.info(f"Saved index to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save index to {path}: {e}")
            return False
