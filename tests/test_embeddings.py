"""Test embedding and Pinecone storage."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion import ingest_pdf
from src.embeddings import generate_embeddings, upsert_chunks, query_similar

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "sciencetb.pdf")


def test_embeddings():
    print("=== Testing embedding generation ===")
    texts = ["The Earth is a unique planet.", "Greenhouse effect traps heat."]
    embeddings = generate_embeddings(texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Dimension: {len(embeddings[0])}")
    print()


def test_upsert():
    print("=== Testing upsert to Pinecone ===")
    result = ingest_pdf(PDF_PATH)
    upsert_chunks(result["chunks"])
    print()


def test_query():
    print("=== Testing similarity search ===")
    query = "Why is Earth suitable for life?"
    results = query_similar(query, top_k=3)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score: {r['score']:.4f}) ---")
        print(f"Page: {r['page']} | Section: {r['section']}")
        print(f"Text: {r['text'][:200]}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["embeddings", "upsert", "query", "all"], default="all")
    args = parser.parse_args()

    if args.test == "all":
        test_embeddings()
        test_upsert()
        test_query()
    else:
        {"embeddings": test_embeddings, "upsert": test_upsert, "query": test_query}[args.test]()
