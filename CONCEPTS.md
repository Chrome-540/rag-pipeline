# RAG Concepts & Explanations

## Phase 1 Concepts

### 1. Document Ingestion
**What:** Converting raw PDFs into searchable chunks.

**Our approach:**
- **PyMuPDF4LLM** extracts text as markdown (preserves tables, headings)
- **PyMuPDF (fitz)** extracts embedded images directly from PDF's internal structure — original quality, not screenshots
- **Gemini Vision** generates text descriptions of images so they become searchable
- Images < 5KB skipped (icons, bullets, decorations)

### 2. Chunking Strategy (Phase 1: header+recursive)
**What:** Breaking documents into smaller pieces for search.

**Why not plain recursive splitting?**
- Destroys tables (splits rows across chunks)
- Loses images entirely
- Breaks math equations
- Ignores document structure

**Our two-stage approach:**
1. **MarkdownHeaderTextSplitter** — splits on #, ##, ### (keeps sections logical)
2. **RecursiveCharacterTextSplitter** (512 chars, 50 overlap) — further splits large sections

Tables preserved as markdown. Images become separate chunks with Vision descriptions.

### 3. Embeddings
**What:** Converting text into numbers (vectors) so we can measure similarity.

**Our choice:** Google `gemini-embedding-001` (3072 dimensions)

Each chunk → 3072-dimensional vector. Similar meanings → similar vectors → close in vector space.

### 4. Vector Database (Pinecone)
**What:** Stores vectors and finds the most similar ones to a query.

**Upsert = Update + Insert:**
- ID doesn't exist → inserts
- ID already exists → updates (no duplicates)

Each vector stored with metadata (text, page, section, source).

### 5. Retrieval
**What:** Finding the most relevant chunks for a user's question.

Phase 1: Top-k similarity search + score threshold (≥ 0.3) filtering.

### 6. Generation (RAG)
**What:** Retrieval-Augmented Generation — LLM answers using retrieved context only.

Flow: Question → embed → find similar chunks → build context → send to Gemini → answer with citations.

System prompt enforces: answer ONLY from context, cite page numbers, say "I don't know" when context is insufficient.

---

## Phase 2 Concepts

### 7. BM25 (Keyword Search)
**Q: What is BM25?**

**A:** BM25 = keyword matching with smarts. For each chunk it asks:
1. Does this chunk contain my query words? (term frequency)
2. How rare are those words across all chunks? (inverse document frequency — rare words matter more)
3. How long is this chunk? (shorter chunks with matches score higher)

**Why we need it alongside vector search:**

| Query | Vector wins | BM25 wins |
|-------|-----------|-----------|
| "Why is Earth special?" | Semantic match | Too generic |
| "Table 13.2" | Might miss (no semantic meaning) | Exact keyword match |
| "greenhouse effect Venus" | Finds related context | Also finds exact mentions |

Vector understands meaning but misses exact terms. BM25 catches exact matches but misses meaning. Together = hybrid search.

### 8. Reciprocal Rank Fusion (RRF)
**Q: How do you merge two ranked lists with different score scales?**

**A:** RRF ignores raw scores and uses rank position only:
```
RRF score = 1/(k + rank)    where k=60 (standard)
```

Example for Chunk A:
- Vector rank 1 → 1/(60+1) = 0.0164
- BM25 rank 2 → 1/(60+2) = 0.0161
- Combined = 0.0325 (appears in both → boosted)

Chunk only in one list → lower score. Chunks both methods agree on → rise to top.

### 9. Image Handling in RAG
**Q: How are images handled? Are they stored?**

**A:**
1. PyMuPDF extracts original embedded images from PDF
2. Gemini Vision generates text description per image
3. Description becomes a chunk: `[Image Description]: ...`
4. Gets embedded and stored in Pinecone like any text chunk
5. Descriptions cached locally in `cache/image_descriptions.json` — Gemini Vision called only once per image

When querying, Pinecone doesn't distinguish image chunks from text chunks — just matches by similarity.

### 10. Persistence & Caching
**Q: Is the data persistent? Do I need to re-ingest?**

**A:** Yes, persistent. Once upserted to Pinecone, data stays permanently.

Caches:
- `cache/image_descriptions.json` — Gemini Vision results (avoids re-calling API)
- `cache/bm25_index.json` — BM25 keyword index (local, rebuilt on ingest)
- Pinecone index object cached in memory (avoids repeated `list_indexes()` calls)

Only re-ingest if: uploading new PDF, or updating existing one, or recreating Pinecone index.

### 11. Parent-Child Chunking
**Q: What is parent-child chunking?**

**A:** Chunk size is a tradeoff:
- Small chunks (256 chars) → precise search, but too little context for LLM
- Large chunks (1024 chars) → good context, but noisy search

Parent-child solves both:
```
Parent (1024 chars) — full section, sent to LLM
├── Child 1 (256 chars) — stored in Pinecone for search
├── Child 2 (256 chars) — stored in Pinecone for search
└── Child 3 (256 chars) — stored in Pinecone for search
```

Small child matches the query precisely → parent gives LLM full context.

**Q: Does it use LLM calls?**

**A:** No. It's purely a chunking strategy — just splitting text differently. Zero API calls.

**Q: Does it modify the existing index?**

**A:** No. Uses a separate Pinecone namespace (`parent-child`). Phase 1 data stays in `default` namespace.
Parents stored locally in `cache/parent_chunks.json`.

---

## Artifacts (Full Visibility)

Every ingestion saves 4 artifacts for inspection:
```
artifacts/
├── sciencetb_1_raw_markdown.json    # Raw text per page
├── sciencetb_2_image_chunks.json    # Image descriptions
├── sciencetb_3_text_chunks.json     # Chunks after splitting (with strategy tag)
└── sciencetb_4_final_chunks.json    # Deduplicated, ready for Pinecone
```

Each chunk has `"strategy": "header+recursive"` in metadata — ready for Phase 2 comparison.
