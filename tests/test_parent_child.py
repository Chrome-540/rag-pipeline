"""Test parent-child chunking strategy."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion import extract_markdown_from_pdf, chunk_parent_child, chunk_markdown
from src.embeddings import upsert_chunks, query_similar
from src.parent_store import save_parents, get_parent_text

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "sciencetb.pdf")


def test_compare_chunking():
    """Compare header+recursive vs parent-child chunks."""
    print("=== Extracting markdown ===")
    md_pages = extract_markdown_from_pdf(PDF_PATH)

    print("\n=== Phase 1: header+recursive ===")
    phase1_chunks = chunk_markdown(md_pages, "sciencetb.pdf")
    print(f"Chunks: {len(phase1_chunks)}")
    print(f"Avg size: {sum(len(c['text']) for c in phase1_chunks) // len(phase1_chunks)} chars")

    print("\n=== Phase 2: parent-child ===")
    result = chunk_parent_child(md_pages, "sciencetb.pdf")
    parents = result["parents"]
    children = result["children"]
    print(f"Parents: {len(parents)}")
    print(f"Children: {len(children)}")
    print(f"Avg parent size: {sum(len(p['text']) for p in parents.values()) // len(parents)} chars")
    print(f"Avg child size: {sum(len(c['text']) for c in children) // len(children)} chars")

    print("\n=== Sample parent-child pair ===")
    sample_child = children[5]
    parent_id = sample_child["metadata"]["parent_id"]
    parent_text = parents[parent_id]["text"]
    print(f"Child ({len(sample_child['text'])} chars): {sample_child['text'][:150]}...")
    print(f"Parent ({len(parent_text)} chars): {parent_text[:300]}...")


def test_upsert_and_query():
    """Upsert children to Pinecone namespace and query with parent lookup."""
    print("=== Building parent-child index ===")
    md_pages = extract_markdown_from_pdf(PDF_PATH)
    result = chunk_parent_child(md_pages, "sciencetb.pdf")

    # Save parents locally
    save_parents(result["parents"])

    # Upsert children to separate namespace
    print("\n=== Upserting children to 'parent-child' namespace ===")
    upsert_chunks(result["children"], namespace="parent-child")

    # Query
    print("\n=== Querying ===")
    query = "What is the habitable zone?"
    results = query_similar(query, top_k=3, namespace="parent-child")

    for i, r in enumerate(results, 1):
        parent_id = r.get("parent_id", "")
        parent_text = get_parent_text(parent_id) if parent_id else ""
        print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"Child: {r['text'][:150]}...")
        if parent_text:
            print(f"Parent: {parent_text[:300]}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["compare", "upsert", "all"], default="compare")
    args = parser.parse_args()

    if args.test == "all":
        test_compare_chunking()
        test_upsert_and_query()
    else:
        {"compare": test_compare_chunking, "upsert": test_upsert_and_query}[args.test]()