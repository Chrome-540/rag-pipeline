# Production RAG Pipeline

A multi-phase Retrieval-Augmented Generation system with multimodal ingestion, dual chunking strategies, hybrid retrieval, automated RAGAS evaluation, and query tracing.

> **[Interactive Showcase](https://rag-spotlight-showcase.lovable.app)** | Built across 3 phases with increasing sophistication

---

## Architecture

```
PDF Upload
     │
     ▼
PyMuPDF Parsing (structured markdown, preserves tables + headers)
     │
     ├──────────────────────────┐
     ▼                          ▼
Text Extraction           Image Extraction
     │                          │
     ▼                          ▼
Markdown Header           Gemini 2.5 Flash Vision
Splitting (#, ##, ###)    (describes graphs, diagrams,
     │                     tables with labels/trends)
     │                          │
     ├──────────────────────────┘
     ▼
Dual Chunking (two Pinecone namespaces)
     ├─ Default: Header-aware + Recursive (512 tokens)
     └─ Parent-Child: Small children (256) search → large parents (1024) returned
     │
     ▼
Deduplication (content hash) → Gemini Embeddings (3072-dim)
     │
     ▼
Dual Indexing
     ├─ Pinecone (cosine similarity, two namespaces)
     └─ BM25 Keyword Index
     │
     ▼
Hybrid Retrieval: Vector + BM25 → RRF Fusion → Cross-Encoder Reranker (optional)
     │
     ▼
Query Tracing (per-phase latency) → Gemini 2.0 Flash → Cited Answer [Page X]
```

---

## Key Features

### Multimodal Ingestion
- PDFs parsed into structured markdown preserving tables, headers, and document hierarchy
- Images and diagrams extracted via PyMuPDF, described by **Gemini 2.5 Flash Vision** (axes, trends, labels)
- Image descriptions cached to avoid redundant API calls on re-ingestion
- Every pipeline stage saves inspection artifacts as JSON

### Dual Chunking Strategies
| Strategy | Namespace | Chunk Size | Purpose |
|----------|-----------|------------|---------|
| Header-aware + Recursive | `default` | 512 tokens, 50 overlap | Respects document structure (chapter → section → subsection) |
| Parent-Child | `parent-child` | Children: 256, Parents: 1024 | Small chunks for precise search, large parents for LLM context |

### 4-Stage Hybrid Retrieval
1. **Vector Search** — Pinecone cosine similarity (Gemini embeddings, 3072-dim)
2. **BM25 Keyword Search** — Exact matches for table refs like "Table 13.2" that embeddings miss
3. **Reciprocal Rank Fusion** — Merges both lists: `score = Σ 1/(k + rank)`
4. **Cross-Encoder Reranker** — Optional second stage using `ms-marco-MiniLM-L-6-v2`. Over-retrieves 20 candidates, returns top-k. Toggle via `RERANK_ENABLED` env var

### Automated RAGAS Evaluation
- 20-question curated golden test set (easy, medium, hard, out-of-scope)
- Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Custom **Page Accuracy** metric: overlap between expected and retrieved source pages

### Query Tracing
- Every request instrumented with per-phase latency (retrieval ms, generation ms)
- Chunk metadata captured (page, score, retrieval method, text preview)
- Traces saved as structured JSON in `traces/`

### Streamlit Dashboard (3 tabs)
- **Chat** — Conversational Q&A with `[Page X]` citations and source list
- **Retrieval Comparison** — Side-by-side: Vector vs BM25 vs Hybrid RRF vs Parent-Child
- **Evaluation** — Aggregate metrics, per-question scores, expected vs generated answers, latency chart

---

## Eval Results

Evaluated on 2 questions (test run). Full 20-question eval pending.

| Metric | Score |
|--------|-------|
| Faithfulness | 1.00 |
| Answer Relevancy | 0.99 |
| Context Precision | 0.99 |
| Context Recall | 1.00 |
| Page Accuracy | 0.75 |

---

## Project Structure

```
rag_pipeline/
├── app.py                  # Streamlit UI (Chat, Comparison, Evaluation tabs)
├── requirements.txt
├── .env                    # API keys (not committed)
│
├── src/
│   ├── main.py             # FastAPI app (/ingest, /query, /query/debug, /documents)
│   ├── ingestion.py        # PDF parsing, image extraction, chunking, dedup
│   ├── embeddings.py       # Gemini embeddings + Pinecone upsert/query
│   ├── bm25.py             # BM25 keyword index
│   ├── retrieval.py        # Hybrid retrieval (vector + BM25 + RRF + reranker)
│   ├── reranker.py         # Cross-encoder reranker (sentence-transformers)
│   ├── generation.py       # Gemini LLM generation with tracing
│   ├── parent_store.py     # Parent chunk cache for parent-child strategy
│   ├── tracer.py           # Query tracing (per-phase latency)
│   ├── config.py           # Settings (Pydantic, env-based)
│   └── logger.py           # Structured logging
│
├── eval/
│   ├── golden_set.json     # 20 curated Q&A pairs with expected pages
│   ├── evaluator.py        # RAGAS evaluation pipeline
│   ├── run_eval.py         # CLI runner
│   └── results.json        # Latest eval results
│
└── tests/
    ├── test_ingestion.py
    ├── test_embeddings.py
    ├── test_retrieval.py
    ├── test_generation.py
    ├── test_bm25.py
    └── test_parent_child.py
```

---

## Setup

### Prerequisites
- Python 3.11+
- [Google AI API key](https://aistudio.google.com/apikey) (Gemini)
- [Pinecone API key](https://www.pinecone.io/)

### Install

```bash
git clone https://github.com/Chrome-540/rag-pipeline.git
cd rag-pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-index
RERANK_ENABLED=false
```

Set `RERANK_ENABLED=true` after installing `sentence-transformers` for cross-encoder reranking.

### Run

```bash
# Start the API
uvicorn src.main:app --reload

# Start the Streamlit UI (separate terminal)
streamlit run app.py

# Run evaluation
python eval/run_eval.py           # Full golden set (20 questions)
python eval/run_eval.py --test 2  # Quick test (2 questions)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Upload and ingest a PDF |
| `POST` | `/query` | Ask a question, get cited answer |
| `POST` | `/query/debug` | Detailed retrieval comparison (vector, BM25, hybrid, parent-child) |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/documents/{filename}` | Delete a document |
| `GET` | `/health` | Health check |

---

## Design Decisions

| Decision | Reasoning |
|----------|-----------|
| Hybrid over pure vector | BM25 catches exact keyword matches (tables, figures) that embeddings miss |
| Two chunking strategies | Header-aware respects structure; parent-child solves precision-vs-context tradeoff |
| Multimodal ingestion | Science textbooks have diagrams and graphs — text-only extraction loses information |
| RAGAS automated eval | Vibe-checking doesn't scale; need quantitative quality gates with a curated test set |
| Cross-encoder reranking | Two-stage: fast recall with bi-encoder, then precise scoring with cross-encoder |
| Query tracing | Production observability: per-phase latency, chunk metadata, debug slow queries |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.0 Flash |
| Vision | Google Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 001 (3072-dim) |
| Vector Store | Pinecone (serverless, cosine) |
| Keyword Search | rank-bm25 |
| Reranker | sentence-transformers (ms-marco-MiniLM-L-6-v2) |
| PDF Parsing | PyMuPDF + PyMuPDF4LLM |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Evaluation | RAGAS |
| Config | Pydantic Settings (.env) |

---

## Phases

- **Phase 1** — Core pipeline: ingestion, embeddings, vector search, generation, Streamlit chat
- **Phase 2** — Advanced retrieval: BM25, RRF fusion, parent-child chunking, reranker, comparison UI
- **Phase 3** — Eval & monitoring: golden test set, RAGAS evaluation, query tracing, eval dashboard

See [PHASE1_DOC.md](PHASE1_DOC.md), [PHASE2_DOC.md](PHASE2_DOC.md), [PHASE3_DOC.md](PHASE3_DOC.md) for detailed step-by-step docs.
