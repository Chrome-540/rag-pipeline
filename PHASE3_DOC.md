# Production RAG — Phase 3: Eval & Monitoring

## Goal
Measure RAG quality, trace every query, and visualize performance.

## Status: COMPLETE

| # | Step | Status |
|---|------|--------|
| 1 | Golden test set (curated Q&A pairs) | Done |
| 2 | RAGAS eval metrics | Done |
| 3 | Query tracing (full pipeline per request) | Done |
| 4 | Eval dashboard in Streamlit | Done |

---

## Step Details

### 1. Golden Test Set
- 20-30 curated Q&A pairs from the textbook
- Mix of easy, medium, hard queries
- Include expected answer + expected source pages
- Used as ground truth for eval

### 2. RAGAS Eval Metrics
- Faithfulness: is the answer grounded in the context?
- Answer relevancy: does the answer address the question?
- Context precision: are the retrieved chunks relevant?
- Context recall: did we retrieve all necessary chunks?
- Automated eval pipeline

### 3. Query Tracing
- Log full pipeline per request: retrieval time, generation time, chunks used, scores
- Save traces to JSON for analysis
- Track latency breakdown

### 4. Eval Dashboard
- Streamlit tab showing eval results
- Per-question scores
- Aggregate metrics
- Latency breakdown
