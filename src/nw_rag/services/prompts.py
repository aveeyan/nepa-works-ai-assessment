# src/nw_rag/services/prompts.py

# RAG System Prompts
RAG_SYSTEM_PROMPT = """You are a precise assistant for Aperture Science documentation.
Answer using ONLY the provided context. If the answer is not in the context, say:
"Insufficient information in the knowledge base. Please check your query or consult the documentation directly."

Always cite the source document.

Context:
{context}"""

RAG_USER_PROMPT = "{question}"

# Fallback Messages
FALLBACK_INSUFFICIENT_CONTEXT = "Insufficient information in the knowledge base. Please check your query or consult the documentation directly."
FALLBACK_LOW_CONFIDENCE = "Insufficient confidence ({score:.3f} < {threshold:.3f}). Please refine your question."
FALLBACK_NO_DOCUMENTS = "No relevant documents found. Please refine your question."
FALLBACK_INJECTION_DETECTED = "Query rejected: potential prompt injection detected."

# Prompt Injection Patterns
INJECTION_PATTERNS = [
    r"ignore.*(?:instructions|previous|above)",
    r"system:\s*",
    r"forget.*(?:all|everything)",
    r"new instruction",
    r"you are now",
    r"as an AI",
    r"override",
    r"bypass"
]
