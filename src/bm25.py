import os
import json
import re
from rank_bm25 import BM25Okapi
from src.logger import get_logger

log = get_logger("bm25")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
BM25_INDEX_FILE = os.path.join(CACHE_DIR, "bm25_index.json")

_bm25_instance = None
_bm25_chunks = None


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.]", " ", text)
    return text.split()


def build_bm25_index(chunks: list[dict]):
    """Build BM25 index from chunks and cache it locally."""
    global _bm25_instance, _bm25_chunks
    log.info(f">> build_bm25_index | input: {len(chunks)} chunks")

    _bm25_chunks = chunks
    tokenized = [_tokenize(c["text"]) for c in chunks]
    _bm25_instance = BM25Okapi(tokenized)

    # Save chunks for reload
    with open(BM25_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    log.info(f"<< build_bm25_index | output: index built, saved to {BM25_INDEX_FILE}")


def load_bm25_index():
    """Load BM25 index from cache."""
    global _bm25_instance, _bm25_chunks
    if _bm25_instance is not None:
        return

    log.info(f">> load_bm25_index | loading from cache")
    if not os.path.exists(BM25_INDEX_FILE):
        log.info(f"<< load_bm25_index | no cache found")
        return

    with open(BM25_INDEX_FILE, "r", encoding="utf-8") as f:
        _bm25_chunks = json.load(f)

    tokenized = [_tokenize(c["text"]) for c in _bm25_chunks]
    _bm25_instance = BM25Okapi(tokenized)
    log.info(f"<< load_bm25_index | loaded {len(_bm25_chunks)} chunks")


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """Search using BM25 keyword matching."""
    log.info(f">> bm25_search | input: query='{query[:50]}...', top_k={top_k}")
    load_bm25_index()

    if _bm25_instance is None or _bm25_chunks is None:
        log.info(f"<< bm25_search | output: no index available")
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25_instance.get_scores(tokenized_query)

    # Get top-k indices sorted by score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, score in ranked:
        if score <= 0:
            continue
        chunk = _bm25_chunks[idx]
        results.append({
            "text": chunk["text"],
            "score": float(score),
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "section": chunk["metadata"].get("section", ""),
            "retrieval": "bm25",
        })

    log.info(f"<< bm25_search | output: {len(results)} results, top_score={results[0]['score']:.3f}" if results else "<< bm25_search | output: 0 results")
    return results
