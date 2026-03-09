"""Test BM25 keyword search."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion import ingest_pdf
from src.bm25 import build_bm25_index, bm25_search

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "sciencetb.pdf")


def test_build_and_search():
    print("=== Building BM25 index ===")
    result = ingest_pdf(PDF_PATH)
    build_bm25_index(result["chunks"])

    queries = [
        "Table 13.2",
        "greenhouse effect",
        "Venus hottest planet",
        "habitable zone",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        results = bm25_search(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. score={r['score']:.3f} | page={r['page']} | {r['text'][:100]}...")


def test_search_only():
    """Search using cached index (no re-ingestion)."""
    queries = [
        "Table 13.2",
        "greenhouse effect",
        "Venus hottest planet",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        results = bm25_search(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. score={r['score']:.3f} | page={r['page']} | {r['text'][:100]}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["build", "search"], default="search")
    args = parser.parse_args()

    {"build": test_build_and_search, "search": test_search_only}[args.test]()
