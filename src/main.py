import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.ingestion import ingest_pdf
from src.embeddings import upsert_chunks, get_or_create_index, query_similar
from src.bm25 import bm25_search
from src.retrieval import retrieve, build_context, get_sources, reciprocal_rank_fusion
from src.generation import generate_answer
from src.parent_store import get_parent_text
from src.logger import get_logger

log = get_logger("api")

app = FastAPI(title="RAG Pipeline", version="0.1.0")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF, parse it, embed it, store in Pinecone."""
    log.info(f">> /ingest | input: {file.filename}")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        log.info(f"<< /ingest | rejected: invalid file type {ext}")
        raise HTTPException(400, f"Only {ALLOWED_EXTENSIONS} files allowed.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        log.info(f"<< /ingest | rejected: file too large ({len(contents)} bytes)")
        raise HTTPException(400, f"File too large. Max {MAX_FILE_SIZE // (1024*1024)}MB.")

    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(contents)
    log.info(f"   saved to {filepath}")

    try:
        result = ingest_pdf(filepath)
        upsert_chunks(result["chunks"])
    except Exception as e:
        log.error(f"<< /ingest | error: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")

    log.info(f"<< /ingest | output: {result['total_chunks']} chunks ingested")
    return {
        "message": f"Ingested {file.filename}",
        "pages": result["total_pages"],
        "text_chunks": result["text_chunks"],
        "image_chunks": result["image_chunks"],
        "total_chunks": result["total_chunks"],
    }


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(req: QueryRequest):
    """Ask a question against ingested documents."""
    log.info(f">> /query | input: '{req.question[:50]}...'")

    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    try:
        result = generate_answer(req.question)
    except Exception as e:
        log.error(f"<< /query | error: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")

    log.info(f"<< /query | output: {len(result['answer'])} chars")
    return result


@app.post("/query/debug")
def query_debug(req: QueryRequest):
    """Return detailed retrieval info for comparison."""
    log.info(f">> /query/debug | input: '{req.question[:50]}...'")

    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    try:
        # Vector-only results
        vector_results = query_similar(req.question, top_k=5)

        # BM25-only results
        bm25_results = bm25_search(req.question, top_k=5)

        # Hybrid (RRF merged)
        hybrid_results = retrieve(req.question, top_k=5)

        # Parent-child results
        pc_results = query_similar(req.question, top_k=5, namespace="parent-child")
        for r in pc_results:
            parent_id = r.get("parent_id", "")
            r["parent_text"] = get_parent_text(parent_id) if parent_id else ""

        # Generate answer using hybrid
        answer_result = generate_answer(req.question)

    except Exception as e:
        log.error(f"<< /query/debug | error: {e}")
        raise HTTPException(500, f"Debug query failed: {str(e)}")

    log.info(f"<< /query/debug | output: done")
    return {
        "answer": answer_result["answer"],
        "sources": answer_result["sources"],
        "debug": {
            "vector": [{"text": r["text"][:200], "score": r["score"], "page": r["page"], "retrieval": "vector"} for r in vector_results],
            "bm25": [{"text": r["text"][:200], "score": r["score"], "page": r["page"], "retrieval": "bm25"} for r in bm25_results],
            "hybrid": [{"text": r["text"][:200], "score": r["score"], "page": r["page"], "retrieval": r.get("retrieval", "")} for r in hybrid_results],
            "parent_child": [{"child": r["text"][:200], "parent": r.get("parent_text", "")[:300], "score": r["score"], "page": r["page"]} for r in pc_results],
        },
    }


@app.get("/documents")
def list_documents():
    """List all uploaded documents."""
    log.info(f">> /documents | listing")
    files = []
    for f in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, f)
        files.append({
            "filename": f,
            "size_kb": round(os.path.getsize(path) / 1024, 1),
        })
    log.info(f"<< /documents | output: {len(files)} files")
    return {"documents": files}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete an uploaded document."""
    log.info(f">> /documents/delete | input: {filename}")
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, "Document not found.")

    os.remove(filepath)
    log.info(f"<< /documents/delete | output: deleted {filename}")
    return {"message": f"Deleted {filename}"}
