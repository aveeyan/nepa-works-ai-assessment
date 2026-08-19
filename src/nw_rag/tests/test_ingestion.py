# src/nw_rag/tests/test_ingestion.py
#
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nw_rag.core.ingestion import DocumentIngester

def test_ingestion():
    ingester = DocumentIngester()

    print("Loading and chunking documents...")
    count = ingester.ingest("./docs")
    print(f"Ingested {count} chunks")

    print("\nSaving index...")
    ingester.save_index()
    print("Index saved")

    print("\nTesting retrieval...")
    results = ingester.retrieve("What is the portal gun?")
    print(f"Found {len(results)} results")

    for doc, score in results[:2]:
        print(f"\nScore: {score:.3f}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
        print("-" * 60)

if __name__ == "__main__":
    test_ingestion()
