"""Test retrieval pipeline."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval import retrieve, build_context, get_sources


def test_retrieve():
    query = "Why is Venus the hottest planet?"
    print(f"Query: {query}\n")

    results = retrieve(query, top_k=5)
    print(f"Results after filtering: {len(results)}\n")

    context = build_context(results)
    print("=== Context for LLM ===")
    print(context[:500])
    print()

    sources = get_sources(results)
    print("=== Sources ===")
    for s in sources:
        print(f"  {s['source']} - Page {s['page']} - {s['section']}")


if __name__ == "__main__":
    test_retrieve()
