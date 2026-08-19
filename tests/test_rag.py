import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.nw_rag.core.ingestion import DocumentIngester
from src.nw_rag.services.rag_engine import RAGEngine

def test_rag():
    print("=" * 60)
    print("Testing RAG Engine")
    print("=" * 60)

    # Initialize ingester
    print("\n1. Initializing ingester...")
    ingester = DocumentIngester()

    # Load existing index
    print("2. Loading index...")
    if not ingester.load_index():
        print("   Index not found. Building...")
        count = ingester.ingest("./docs")
        ingester.save_index()
        print(f"   Built index with {count} chunks")
    else:
        print("   Index loaded successfully")

    # Initialize RAG engine
    print("3. Initializing RAG engine...")
    rag = RAGEngine(ingester)

    # Test questions
    test_questions = [
        "What is the portal gun?",
        "How do I handle a weighted storage cube?",
        "What are the emergency procedures?",
        "Tell me about GLaDOS",
        "What is the cake recipe?"
    ]

    print("\n4. Running queries...")
    print("-" * 60)

    for question in test_questions:
        print(f"\nQ: {question}")
        print("-" * 40)

        result = rag.query(question)

        print(f"A: {result['answer'][:300]}")
        if len(result['answer']) > 300:
            print("...")

        print(f"\nConfidence: {result['confidence']:.3f}")
        print(f"Fallback: {result['fallback']}")
        print(f"Sources: {len(result['sources'])} found")
        print(f"Retrieval time: {result['timing']['retrieval']:.3f}s")
        print(f"Generation time: {result['timing']['generation']:.3f}s")

        if result['sources']:
            print("Sources:")
            for src in result['sources'][:2]:
                print(f"  - {src['source']} (score: {src['score']:.3f})")

        print("-" * 40)

if __name__ == "__main__":
    test_rag()
