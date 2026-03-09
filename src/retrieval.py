from src.embeddings import query_similar
from src.config import settings
from src.logger import get_logger

log = get_logger("retrieval")


def retrieve(query: str, top_k: int = None, score_threshold: float = 0.3) -> list[dict]:
    """Retrieve relevant chunks, filter by score threshold."""
    log.info(f">> retrieve | input: query='{query[:50]}...', threshold={score_threshold}")
    results = query_similar(query, top_k=top_k or settings.top_k)

    # Filter out low-confidence results
    filtered = [r for r in results if r["score"] >= score_threshold]

    for i, r in enumerate(filtered, 1):
        log.info(f"   chunk {i} | score={r['score']:.3f} | page={r['page']} | {r['text'][:100]}...")

    log.info(f"<< retrieve | output: {len(filtered)}/{len(results)} chunks passed threshold")
    return filtered


def build_context(results: list[dict]) -> str:
    """Assemble retrieved chunks into a formatted context string for the LLM."""
    log.info(f">> build_context | input: {len(results)} chunks")
    if not results:
        log.info(f"<< build_context | output: empty context")
        return ""

    context_parts = []
    for i, r in enumerate(results, 1):
        source_info = f"[Source: {r['source']}, Page {r['page']}]"
        section = f" | Section: {r['section']}" if r.get("section") else ""
        context_parts.append(
            f"--- Chunk {i} {source_info}{section} (score: {r['score']:.3f}) ---\n{r['text']}"
        )

    context = "\n\n".join(context_parts)
    log.info(f"<< build_context | output: {len(context)} chars")
    return context


def get_sources(results: list[dict]) -> list[dict]:
    """Extract unique sources from results for citation."""
    log.info(f">> get_sources | input: {len(results)} results")
    seen = set()
    sources = []
    for r in results:
        key = (r["source"], r["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": r["source"],
                "page": r["page"],
                "section": r.get("section", ""),
            })
    log.info(f"<< get_sources | output: {len(sources)} unique sources")
    return sources
