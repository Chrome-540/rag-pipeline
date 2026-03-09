from google import genai
from src.config import settings
from src.retrieval import retrieve, build_context, get_sources
from src.logger import get_logger

log = get_logger("generation")

client = genai.Client(api_key=settings.google_api_key)

SYSTEM_PROMPT = """You are a helpful science tutor. Answer questions based ONLY on the provided context.
Rules:
- If the context doesn't contain enough information, say "I don't have enough information to answer this."
- Cite the source page numbers in your answer like [Page X].
- Keep answers clear and concise.
- If the context contains table data, present it neatly."""


def generate_answer(query: str) -> dict:
    """Full RAG pipeline: retrieve context, generate answer with citations."""
    log.info(f">> generate_answer | input: query='{query[:50]}...'")

    # 1. Retrieve relevant chunks
    results = retrieve(query)

    if not results:
        log.info(f"<< generate_answer | output: no relevant chunks found")
        return {
            "answer": "I don't have enough information to answer this.",
            "sources": [],
        }

    # 2. Build context
    context = build_context(results)

    # 3. Generate answer
    prompt = f"""Context:
{context}

Question: {query}

Answer the question using only the context above. Cite page numbers."""

    log.info(f"   calling Gemini with {len(context)} chars of context")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={"system_instruction": SYSTEM_PROMPT},
    )

    # 4. Extract sources
    sources = get_sources(results)

    log.info(f"<< generate_answer | output: {len(response.text)} chars, {len(sources)} sources")
    return {
        "answer": response.text,
        "sources": sources,
    }
