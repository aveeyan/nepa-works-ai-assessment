# src/nw_rag/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    SEPARATORS = ["\n\n", "\n", ".", " ", ""]
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))
    TOP_K = int(os.getenv("TOP_K", 3))
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
    ALLOW_DANGEROUS_DESERIALIZATION = True
