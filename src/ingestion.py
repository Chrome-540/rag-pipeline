import os
import json
import hashlib
import pymupdf4llm
import fitz
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from src.config import settings
from src.logger import get_logger

log = get_logger("ingestion")

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
IMAGE_CACHE_FILE = os.path.join(CACHE_DIR, "image_descriptions.json")


def _save_artifact(name: str, data):
    """Save a pipeline artifact as JSON for inspection."""
    path = os.path.join(ARTIFACTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"   artifact saved: artifacts/{name}")


def _load_image_cache() -> dict:
    if os.path.exists(IMAGE_CACHE_FILE):
        with open(IMAGE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_image_cache(cache: dict):
    with open(IMAGE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def extract_markdown_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text as markdown per page using PyMuPDF4LLM. Preserves tables and structure."""
    log.info(f">> extract_markdown_from_pdf | input: {pdf_path}")
    md_pages = pymupdf4llm.to_markdown(pdf_path, pages=None, page_chunks=True)
    log.info(f"<< extract_markdown_from_pdf | output: {len(md_pages)} pages")
    return md_pages


def extract_images_from_pdf(pdf_path: str, output_dir: str) -> list[dict]:
    """Extract images from PDF and save them with metadata."""
    log.info(f">> extract_images_from_pdf | input: {pdf_path}")
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            # Skip tiny images (likely icons/decorations)
            if len(image_bytes) < 5000:
                log.info(f"   skipping small image on page {page_num+1} ({len(image_bytes)} bytes)")
                continue

            filename = f"page{page_num + 1}_img{img_idx + 1}.{ext}"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "wb") as f:
                f.write(image_bytes)

            images.append({
                "path": filepath,
                "page": page_num + 1,
                "index": img_idx + 1,
            })

    doc.close()
    log.info(f"<< extract_images_from_pdf | output: {len(images)} images extracted")
    return images


def describe_image(image_path: str) -> str:
    """Use Gemini Vision to generate a text description of an image."""
    log.info(f">> describe_image | input: {image_path}")
    from google import genai
    from PIL import Image

    client = genai.Client(api_key=settings.google_api_key)
    img = Image.open(image_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "Describe this image from a science textbook in detail. "
            "Include any data, labels, or key information visible. if its image of a boy or girl, just ignore it and say its a image of a person. If its a graph, describe the axes, trends, and any annotations.",
            img,
        ],
    )
    log.info(f"<< describe_image | output: {len(response.text)} chars")
    return response.text


def chunk_markdown(md_pages: list[dict], source_filename: str) -> list[dict]:
    """Chunk markdown pages by headers first, then by size."""
    log.info(f">> chunk_markdown | input: {len(md_pages)} pages, source={source_filename}")

    # Split by markdown headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "chapter"),
            ("##", "section"),
            ("###", "subsection"),
        ]
    )

    # Further split large sections by size
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    all_chunks = []
    chunk_idx = 0

    for page_data in md_pages:
        page_num = page_data.get("metadata", {}).get("page", 0)
        text = page_data.get("text", "")

        if not text.strip():
            continue

        # First split by headers
        header_chunks = header_splitter.split_text(text)

        # Then split large chunks by size
        for hchunk in header_chunks:
            sub_chunks = size_splitter.split_text(hchunk.page_content)

            for sub in sub_chunks:
                content_hash = hashlib.md5(sub.encode()).hexdigest()
                all_chunks.append({
                    "text": sub,
                    "metadata": {
                        "source": source_filename,
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "chapter": hchunk.metadata.get("chapter", ""),
                        "section": hchunk.metadata.get("section", ""),
                        "subsection": hchunk.metadata.get("subsection", ""),
                        "content_hash": content_hash,
                        "strategy": "header+recursive",
                    },
                })
                chunk_idx += 1

    log.info(f"<< chunk_markdown | output: {len(all_chunks)} chunks")
    return all_chunks


def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks based on content hash."""
    log.info(f">> deduplicate_chunks | input: {len(chunks)} chunks")
    seen = set()
    unique = []
    for chunk in chunks:
        h = chunk["metadata"]["content_hash"]
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    removed = len(chunks) - len(unique)
    log.info(f"<< deduplicate_chunks | output: {len(unique)} unique, {removed} duplicates removed")
    return unique


def ingest_pdf(pdf_path: str) -> dict:
    """Full ingestion pipeline for a PDF file."""
    log.info(f">> ingest_pdf | input: {pdf_path}")
    filename = os.path.basename(pdf_path)
    stem = os.path.splitext(filename)[0]
    images_dir = os.path.join(os.path.dirname(pdf_path), "extracted_images")

    # 1. Extract markdown
    md_pages = extract_markdown_from_pdf(pdf_path)
    _save_artifact(f"{stem}_1_raw_markdown.json", [
        {"page": p.get("metadata", {}).get("page", i), "text": p.get("text", "")}
        for i, p in enumerate(md_pages)
    ])

    # 2. Extract and describe images (with cache)
    images = extract_images_from_pdf(pdf_path, images_dir)
    image_cache = _load_image_cache()
    image_chunks = []
    for img in images:
        try:
            cache_key = os.path.basename(img["path"])
            if cache_key in image_cache:
                description = image_cache[cache_key]
                log.info(f"   cache hit for {cache_key}")
            else:
                description = describe_image(img["path"])
                image_cache[cache_key] = description
                _save_image_cache(image_cache)
            content_hash = hashlib.md5(description.encode()).hexdigest()
            image_chunks.append({
                "text": f"[Image Description]: {description}",
                "metadata": {
                    "source": filename,
                    "page": img["page"],
                    "chunk_index": -1,
                    "chapter": "",
                    "section": "",
                    "subsection": "",
                    "content_hash": content_hash,
                    "type": "image",
                },
            })
        except Exception as e:
            log.error(f"   failed to describe image {img['path']}: {e}")

    _save_artifact(f"{stem}_2_image_chunks.json", image_chunks)

    # 3. Chunk text
    text_chunks = chunk_markdown(md_pages, filename)
    _save_artifact(f"{stem}_3_text_chunks.json", text_chunks)

    # 4. Combine and deduplicate
    all_chunks = deduplicate_chunks(text_chunks + image_chunks)
    _save_artifact(f"{stem}_4_final_chunks.json", all_chunks)

    log.info(f"<< ingest_pdf | output: {len(all_chunks)} total chunks (text={len(text_chunks)}, images={len(image_chunks)})")

    return {
        "filename": filename,
        "total_pages": len(md_pages),
        "text_chunks": len(text_chunks),
        "image_chunks": len(image_chunks),
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
    }
