from pinecone import Pinecone, ServerlessSpec
from google import genai
from src.config import settings
from src.logger import get_logger

log = get_logger("embeddings")

# Initialize clients
genai_client = genai.Client(api_key=settings.google_api_key)
pc = Pinecone(api_key=settings.pinecone_api_key)


_index_cache = None


def get_or_create_index():
    """Get existing Pinecone index or create one. Cached after first call."""
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    log.info(f">> get_or_create_index | input: {settings.pinecone_index_name}")
    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        log.info(f"   creating new index: {settings.pinecone_index_name}")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    _index_cache = pc.Index(settings.pinecone_index_name)
    log.info(f"<< get_or_create_index | output: index ready (cached)")
    return _index_cache


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using Google's embedding model."""
    log.info(f">> generate_embeddings | input: {len(texts)} texts")
    response = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )
    embeddings = [e.values for e in response.embeddings]
    log.info(f"<< generate_embeddings | output: {len(embeddings)} embeddings, dim={len(embeddings[0])}")
    return embeddings


def upsert_chunks(chunks: list[dict], batch_size: int = 100):
    """Embed chunks and upsert them into Pinecone."""
    log.info(f">> upsert_chunks | input: {len(chunks)} chunks, batch_size={batch_size}")
    index = get_or_create_index()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = generate_embeddings(texts)

        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            vector_id = chunk["metadata"]["content_hash"]
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "chapter": chunk["metadata"].get("chapter", ""),
                    "section": chunk["metadata"].get("section", ""),
                    "subsection": chunk["metadata"].get("subsection", ""),
                },
            })

        index.upsert(vectors=vectors)
        log.info(f"   upserted batch {i // batch_size + 1} ({len(vectors)} vectors)")

    log.info(f"<< upsert_chunks | output: {len(chunks)} vectors upserted")


def query_similar(query: str, top_k: int = None) -> list[dict]:
    """Find chunks most similar to the query."""
    top_k = top_k or settings.top_k
    log.info(f">> query_similar | input: query='{query[:50]}...', top_k={top_k}")
    index = get_or_create_index()

    query_embedding = generate_embeddings([query])[0]

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    matches = [
        {
            "text": match.metadata["text"],
            "score": match.score,
            "source": match.metadata["source"],
            "page": match.metadata["page"],
            "section": match.metadata.get("section", ""),
        }
        for match in results.matches
    ]
    log.info(f"<< query_similar | output: {len(matches)} matches, top_score={matches[0]['score']:.3f}" if matches else "<< query_similar | output: 0 matches")
    return matches
