# Production RAG — Phase 2: Advanced Retrieval

## Goal
Improve retrieval quality with hybrid search, reranking, and smarter chunking. Full visibility to compare strategies.

## Status: NOT STARTED

| # | Step | Status |
|---|------|--------|
| 1 | Hybrid search (BM25 + vector) | Done |
| 2 | Reciprocal Rank Fusion (merge results) | Done |
| 3 | Cross-encoder reranker | Not started |
| 4 | Query rewriting (LLM) | Not started |
| 5 | Parent-child chunking | Done |
| 6 | Semantic chunking | Skipped (headers sufficient) |
| 7 | Streamlit strategy comparison UI | Done |

---

## Step Details

### 1. Hybrid Search
- Add BM25 keyword index alongside Pinecone vector search
- BM25 catches exact keyword matches that embeddings might miss
- Example: "Table 13.2" — BM25 finds it exactly, vector search may not

### 2. Reciprocal Rank Fusion (RRF)
- Merge BM25 and vector results into one ranked list
- Formula: score = 1/(k + rank), summed across both lists
- Balances semantic similarity with keyword relevance

### 3. Cross-Encoder Reranker
- Retrieve top-20 candidates → rerank with cross-encoder → return top-5
- Cross-encoder scores (query, chunk) pairs directly — more accurate than embedding similarity
- Options: Cohere Rerank, bge-reranker-v2, or Gemini-based

### 4. Query Rewriting
- LLM rewrites vague/incomplete queries before retrieval
- Example: "what about venus?" → "What makes Venus the hottest planet in the solar system?"
- Improves retrieval recall

### 5. Parent-Child Chunking
- Small chunks (256 tokens) for search precision
- When a small chunk matches, return its parent section for full context
- Avoids the chunk-size tradeoff (small = precise search, large = better context)

### 6. Semantic Chunking
- Split by topic boundaries instead of fixed character count
- Uses embedding similarity between sentences to detect topic shifts
- Compare with Phase 1's header+recursive strategy using artifacts

### 7. Streamlit Strategy Comparison
- Side-by-side view: Phase 1 vs Phase 2 retrieval results
- Show scores, chunks, and final answer for each strategy
- Toggle between chunking strategies

---

## Artifacts (for comparison with Phase 1)
All new strategies will save artifacts in the same format:
```
artifacts/
├── sciencetb_3_text_chunks.json                # Phase 1: header+recursive
├── sciencetb_3_text_chunks_semantic.json        # Phase 2: semantic
├── sciencetb_3_text_chunks_parent_child.json    # Phase 2: parent-child
```

Each chunk's metadata includes `"strategy"` field for filtering and comparison.

---

## New Dependencies (expected)
- `rank-bm25` — BM25 keyword search
- `sentence-transformers` — cross-encoder reranker (or Cohere API)
