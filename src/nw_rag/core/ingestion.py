# src/nw_rag/core/ingestion.py

# Standard Imports
import os
from typing import List, Tuple

# Third-Party Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

# Local Imports
from nw_rag.config import Config  # Changed to relative import

class DocumentIngester:
    """Handles the ingestion of documents, including loading, splitting, and embedding."""
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.EMBEDDING_MODEL
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=Config.SEPARATORS,
            length_function=len
        )
        self.vectorstore = None

    def load_documents(self, docs_dir: str = None) -> List[Document]:
        """Loads documents from the specified directory."""
        docs_dir = docs_dir or Config.DOCS_DIR
        documents = []

        if not os.path.exists(docs_dir):
            raise ValueError(f"Directory {docs_dir} does not exist")

        for file in os.listdir(docs_dir):
            path = os.path.join(docs_dir, file)
            if file.endswith(".txt"):
                loader = TextLoader(path)
            elif file.endswith(".md"):
                loader = UnstructuredMarkdownLoader(path)
            else:
                continue
            documents.extend(loader.load())

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into chunks."""
        return self.splitter.split_documents(documents)

    def ingest(self, docs_dir: str = None) -> int:
        """Ingests documents from the specified directory."""
        docs = self.load_documents(docs_dir)
        chunks = self.chunk_documents(docs)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return len(chunks)

    def save_index(self, path: str = None) -> bool:
        """Saves the vectorstore index to disk."""
        path = path or Config.FAISS_INDEX_PATH
        if self.vectorstore is not None:
            self.vectorstore.save_local(path)
            return True
        return False

    def load_index(self, path: str = None) -> bool:
        """Loads the vectorstore index from disk."""
        path = path or Config.FAISS_INDEX_PATH
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=Config.ALLOW_DANGEROUS_DESERIALIZATION
            )
            return True
        return False

    def retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Retrieves the top-k documents for a given query."""
        k = k or Config.TOP_K
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Run ingest() first.")
        return self.vectorstore.similarity_search_with_score(query, k=k)
