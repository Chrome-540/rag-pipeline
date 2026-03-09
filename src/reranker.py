"""Cross-encoder reranker for second-stage retrieval."""

from sentence_transformers import CrossEncoder
from src.logger import get_logger

log = get_logger("reranker")

_model = None


def _get_model() -> CrossEncoder:
    """Lazy-load the cross-encoder model (cached after first call)."""
    global _model
    if _model is None:
        log.info("Loading cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        log.info("Cross-encoder model loaded")
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank chunks using a cross-encoder model.

    Takes (query, chunk_text) pairs, scores them with a cross-encoder,
    and returns the top_k chunks sorted by relevance.
    """
    if not chunks:
        return []

    log.info(f">> rerank | input: {len(chunks)} chunks, top_k={top_k}")

    model = _get_model()
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = model.predict(pairs)

    # Attach rerank scores and sort
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    final = reranked[:top_k]

    top_scores = [round(c["rerank_score"], 3) for c in final[:3]]
    log.info(f"<< rerank | output: top scores = {top_scores}")
    return final
