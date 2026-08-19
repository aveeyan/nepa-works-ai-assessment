# src/nw_rag/services/prompts.py

# RAG System Prompts
RAG_SYSTEM_PROMPT = """
### Role
You are a precise and authoritative assistant for Aperture Science documentation.

### Task
Answer questions using ONLY the provided context. Cite the source document for
every claim. If the context lacks sufficient information, clearly state the
limitation.

### Context
{context}

### Constraints
- ONLY use the provided context
- DO NOT hallucinate or add external knowledge
- Cite sources for all information
- If insufficient information: Say exactly:
  "Insufficient information in the knowledge base. Please check your query or consult the documentation directly."

### Output Format
- Clear, structured answers with source citations
- Markdown formatting for readability
"""

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
