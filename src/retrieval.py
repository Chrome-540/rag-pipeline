from src.embeddings import query_similar
from src.bm25 import bm25_search
from src.config import settings
from src.logger import get_logger

log = get_logger("retrieval")


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge multiple ranked lists using RRF."""
    log.info(f">> rrf | input: {len(result_lists)} lists, k={k}")

    # Score each chunk by summing 1/(k + rank) across all lists
    rrf_scores = {}  # text_hash -> {score, chunk_data}

    for results in result_lists:
        for rank, r in enumerate(results, 1):
            key = r["text"][:200]  # use text prefix as key
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "chunk": r, "found_in": []}
            rrf_scores[key]["score"] += 1.0 / (k + rank)
            rrf_scores[key]["found_in"].append(r.get("retrieval", "unknown"))

    # Sort by RRF score
    ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    results = []
    for item in ranked:
        chunk = item["chunk"]
        chunk["score"] = item["score"]
        chunk["retrieval"] = "+".join(item["found_in"])
        results.append(chunk)

    log.info(f"<< rrf | output: {len(results)} merged results")
    return results


def retrieve(query: str, top_k: int = None, score_threshold: float = None) -> list[dict]:
    """Hybrid retrieve: vector + BM25, merged with RRF."""
    top_k = top_k or settings.top_k
    log.info(f">> retrieve | input: query='{query[:50]}...', top_k={top_k}")

    # Get results from both sources
    vector_results = query_similar(query, top_k=top_k)
    bm25_results = bm25_search(query, top_k=top_k)

    # Merge with RRF
    merged = reciprocal_rank_fusion([vector_results, bm25_results])

    # Take top_k
    final = merged[:top_k]

    for i, r in enumerate(final, 1):
        log.info(f"   chunk {i} | rrf={r['score']:.4f} | via={r['retrieval']} | page={r['page']} | {r['text'][:100]}...")

    log.info(f"<< retrieve | output: {len(final)} chunks")
    return final


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
            f"--- Chunk {i} {source_info}{section} (score: {r['score']:.4f}, via: {r.get('retrieval', 'unknown')}) ---\n{r['text']}"
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
