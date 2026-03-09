# Production RAG — Phase 1: Foundations

## Goal
Working API + UI: upload PDFs, ask questions, get answers with source citations.

## Status: COMPLETE

| # | Step | Status |
|---|------|--------|
| 1 | Project setup (structure, deps, config) | Done |
| 2 | Document ingestion (loaders, chunking, metadata) | Done |
| 3 | Embedding & storage (Pinecone) | Done |
| 4 | Retrieval (similarity search, filtering) | Done |
| 5 | Generation (prompt template, Gemini, citations) | Done |
| 6 | API layer (FastAPI endpoints) | Done |
| 7 | Basic guardrails | Done |
| 8 | Streamlit frontend | Done |
| 9 | Logging (time, flow, i/o) | Done |
| 10 | Image description caching | Done |
| 11 | Pinecone index caching | Done |

---

## Project Structure

```
rag_pipeline/
├── src/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings, loads .env
│   ├── logger.py          # Logger (console + file at logs/rag.log)
│   ├── ingestion.py       # PDF parsing, image extraction, chunking
│   ├── embeddings.py      # Google embeddings, Pinecone upsert/query
│   ├── retrieval.py       # Similarity search, score filtering, context builder
│   ├── generation.py      # Gemini LLM call, prompt template, citations
│   └── main.py            # FastAPI endpoints
├── tests/
│   ├── test_ingestion.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── cache/
│   └── image_descriptions.json   # Cached Gemini Vision descriptions
├── logs/
│   └── rag.log                   # Persistent log file
├── uploads/                      # Uploaded PDFs stored here
├── extracted_images/             # Images pulled from PDFs
├── app.py                        # Streamlit frontend
├── requirements.txt
└── .env                          # API keys (not committed)
```

---

## Stack

| Component | Choice |
|-----------|--------|
| Language | Python |
| API | FastAPI |
| Frontend | Streamlit |
| Vector DB | Pinecone (serverless, cosine, 3072 dim) |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Google gemini-embedding-001 (3072 dim) |
| Image descriptions | Google Gemini 2.5 Flash (Vision) |
| PDF parsing | PyMuPDF4LLM (markdown output) |
| Chunking | LangChain (header + recursive) |
| Config | pydantic-settings + .env |

---

## How It Works

### Ingestion Flow
```
PDF Upload → PyMuPDF4LLM (text as markdown) → Header-based chunking → Size-based sub-chunking → Deduplication
         └→ PyMuPDF (image extraction) → Gemini Vision (descriptions, cached) → Image chunks
         └→ All chunks → Google Embedding → Pinecone Upsert
```

### Query Flow
```
User Question → Google Embedding → Pinecone similarity search (top-k=10)
            → Score threshold filter (≥0.3) → Context assembly → Gemini LLM → Answer + Citations
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /ingest | Upload PDF, parse, embed, store |
| POST | /query | Ask a question, get answer + sources |
| GET | /documents | List uploaded documents |
| DELETE | /documents/{filename} | Delete a document |

---

## Chunking Strategy

**Two-stage approach:**
1. **Header splitting** — MarkdownHeaderTextSplitter splits on #, ##, ### (preserves sections)
2. **Size splitting** — RecursiveCharacterTextSplitter (512 chars, 50 overlap) for large sections

Tables are preserved as markdown within chunks. Images get separate chunks with Gemini Vision descriptions.

---

## Key Decisions

- **Chunk size:** 512 tokens, 50 overlap
- **Top-k:** 10 results retrieved, filtered by score ≥ 0.3
- **Image threshold:** Skip images < 5KB (icons/decorations)
- **Image cache:** Descriptions saved to cache/image_descriptions.json (avoids repeat Vision API calls)
- **Index cache:** Pinecone index object cached after first connection (avoids repeated list_indexes calls)
- **System prompt:** Answers based ONLY on provided context, with page citations

---

## Logging

All modules log to console + `logs/rag.log` with format:
```
[HH:MM:SS] module | >> function | input: ...
[HH:MM:SS] module |    chunk 1 | score=0.892 | page=4 | text preview...
[HH:MM:SS] module | << function | output: ...
```

---

## How to Run

```bash
# Install deps
pip install -r requirements.txt

# Terminal 1 — API
python -m uvicorn src.main:app --reload

# Terminal 2 — Frontend
streamlit run app.py
```

---

## Test Data
- **Document:** NCERT Science Class 8, Chapter 13 — "Our Home: Earth, a Unique Life Sustaining Planet"
- **Content:** Text, tables (planet data), diagrams (Earth layers, greenhouse effect, habitable zone)
- **Results:** 19 pages, ~93 text chunks + ~63 image chunks

---

## Next Phases

- **Phase 2:** Hybrid search (BM25 + vector), rerankers (cross-encoder), query rewriting, parent-child chunking
- **Phase 3:** RAGAS eval metrics, golden test set, observability (tracing), feedback loop, caching, guardrails
