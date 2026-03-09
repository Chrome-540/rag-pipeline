"""Test the ingestion pipeline with sciencetb.pdf"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion import (
    extract_markdown_from_pdf,
    extract_images_from_pdf,
    chunk_markdown,
    deduplicate_chunks,
    ingest_pdf,
)

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "sciencetb.pdf")


def test_extract_markdown():
    print("=== Testing markdown extraction ===")
    pages = extract_markdown_from_pdf(PDF_PATH)
    print(f"Total pages extracted: {len(pages)}")
    print(f"First page preview:\n{pages[0]['text'][:200]}")
    print()


def test_extract_images():
    print("=== Testing image extraction ===")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "extracted_images")
    images = extract_images_from_pdf(PDF_PATH, out_dir)
    print(f"Images extracted: {len(images)}")
    for img in images[:3]:
        print(f"  Page {img['page']}: {img['path']}")
    print()


def test_chunking():
    print("=== Testing chunking ===")
    pages = extract_markdown_from_pdf(PDF_PATH)
    chunks = chunk_markdown(pages, "sciencetb.pdf")
    chunks = deduplicate_chunks(chunks)
    print(f"Total chunks: {len(chunks)}")
    print(f"\nSample chunk (index 2):")
    if len(chunks) > 2:
        c = chunks[2]
        print(f"  Page: {c['metadata']['page']}")
        print(f"  Section: {c['metadata']['section']}")
        print(f"  Text: {c['text'][:200]}")
    print()


def test_full_pipeline():
    print("=== Testing full ingestion pipeline ===")
    result = ingest_pdf(PDF_PATH)
    print(f"Pages: {result['total_pages']}")
    print(f"Text chunks: {result['text_chunks']}")
    print(f"Image chunks: {result['image_chunks']}")
    print(f"Total chunks: {result['total_chunks']}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["markdown", "images", "chunking", "full"], default="full")
    args = parser.parse_args()

    tests = {
        "markdown": test_extract_markdown,
        "images": test_extract_images,
        "chunking": test_chunking,
        "full": test_full_pipeline,
    }
    tests[args.test]()
