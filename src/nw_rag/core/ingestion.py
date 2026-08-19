# src/nw_rag/core/ingestion.py

# Standard Imports
import os
from typing import List

# Third-Party Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

# Local Imports
from nw_rag.config import Config
from nw_rag.core.retrieval import VectorRetriever
from nw_rag.logging_config import logger


class DocumentIngester:
    """Handles document ingestion and vectorstore creation."""

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
        self.retriever = VectorRetriever()

    def load_documents(self, docs_dir: str = None) -> List[Document]:
        """Load documents from directory."""
        docs_dir = docs_dir or Config.DOCS_DIR
        documents = []

        if not os.path.exists(docs_dir):
            raise ValueError(f"Directory {docs_dir} does not exist")

        for file in os.listdir(docs_dir):
            path = os.path.join(docs_dir, file)
            try:
                if file.endswith(".txt"):
                    loader = TextLoader(path, encoding='utf-8')
                    documents.extend(loader.load())
                    logger.debug(f"Loaded TXT: {file}")

                elif file.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(path)
                    documents.extend(loader.load())
                    logger.debug(f"Loaded MD: {file}")

                elif file.endswith(".pdf"):
                    try:
                        loader = PyPDFLoader(path)
                        docs = loader.load()
                        documents.extend(docs)
                        logger.debug(f"Loaded PDF: {file} ({len(docs)} pages)")
                    except Exception as e:
                        logger.error(f"PyPDFLoader failed for {file}: {e}")
                        # Fallback to pypdf
                        try:
                            from pypdf import PdfReader
                            reader = PdfReader(path)
                            text = ""
                            for page in reader.pages:
                                text += page.extract_text() + "\n"
                            if text.strip():
                                documents.append(Document(
                                    page_content=text,
                                    metadata={"source": path, "file": file}
                                ))
                                logger.debug(f"Loaded PDF via pypdf: {file}")
                            else:
                                logger.warning(f"No text extracted from PDF: {file}")
                        except Exception as e2:
                            logger.error(f"Failed to load PDF {file}: {e2}")
                else:
                    continue

            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
                continue

        logger.info(f"Loaded {len(documents)} documents from {docs_dir}")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def ingest(self, docs_dir: str = None) -> int:
        """Ingest documents and create vectorstore."""
        docs = self.load_documents(docs_dir)

        if not docs:
            raise ValueError("No documents loaded. Check docs directory and file formats.")

        chunks = self.chunk_documents(docs)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.retriever.set_vectorstore(self.vectorstore)

        logger.info(f"Indexed {len(chunks)} chunks")
        return len(chunks)

    def save_index(self, path: str = None) -> bool:
        """Save vectorstore to disk."""
        path = path or Config.FAISS_INDEX_PATH
        return self.retriever.save_index(path)

    def load_index(self, path: str = None) -> bool:
        """Load vectorstore from disk."""
        path = path or Config.FAISS_INDEX_PATH
        success = self.retriever.load_index(path)
        if success:
            self.vectorstore = self.retriever.vectorstore
        return success

    def retrieve(self, query: str, k: int = None) -> List:
        """Retrieve documents for a query."""
        return self.retriever.retrieve(query, k)

    def retrieve_with_threshold(self, query: str, k: int = None, threshold: float = None) -> List:
        """Retrieve documents with threshold filtering."""
        return self.retriever.retrieve_with_threshold(query, k, threshold)

    def get_top_score(self, query: str) -> float:
        """Get highest similarity score for a query."""
        return self.retriever.get_top_score(query)
