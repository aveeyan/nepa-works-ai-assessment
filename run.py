# run.py (at project root)

# Standard Imports
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Third-Party Imports
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "nw_rag.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
