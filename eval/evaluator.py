"""Core RAGAS evaluation logic for the RAG pipeline."""

import json
import asyncio
import os
from pathlib import Path

from google import genai
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)

from src.config import settings
from src.retrieval import retrieve, build_context
from src.generation import generate_answer
from src.logger import get_logger
from src.tracer import Trace

log = get_logger("evaluator")

EVAL_DIR = Path(__file__).parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"
RESULTS_PATH = EVAL_DIR / "results.json"


class GoogleEmbeddingsAdapter:
    """Wraps modern GoogleEmbeddings to add legacy embed_query/embed_documents methods."""

    def __init__(self, google_embeddings):
        self._emb = google_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._emb.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_texts(texts)

    def __getattr__(self, name):
        return getattr(self._emb, name)


def load_golden_set() -> list[dict]:
    """Load the golden test set."""
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_page_accuracy(expected_pages: list[int], retrieved_chunks: list[dict]) -> float:
    """Compute overlap between expected and actually retrieved pages."""
    if not expected_pages:
        # Out-of-scope question: accuracy = 1.0 if no pages retrieved, else 0.0
        return 1.0 if not retrieved_chunks else 0.0

    retrieved_pages = set()
    for chunk in retrieved_chunks:
        page = chunk.get("page")
        if page is not None:
            retrieved_pages.add(int(page))

    expected = set(expected_pages)
    if not expected:
        return 1.0

    overlap = expected & retrieved_pages
    return len(overlap) / len(expected)


def run_pipeline(question: str) -> dict:
    """Run RAG pipeline for a single question, returning all eval data."""
    log.info(f"Running pipeline for: {question[:60]}...")

    trace = Trace(question)

    # Retrieve chunks
    trace.start("retrieval")
    chunks = retrieve(question)
    trace.end("retrieval")

    # Extract context texts
    contexts = [chunk["text"] for chunk in chunks]

    # Generate answer
    trace.start("generation")
    result = generate_answer(question)
    trace.end("generation")
    answer = result["answer"]

    trace.set_chunks(chunks)
    trace_data = trace.save()

    return {
        "answer": answer,
        "contexts": contexts,
        "chunks": chunks,
        "trace": trace_data,
    }


def build_eval_dataset(golden_set: list[dict], limit: int = None) -> tuple:
    """Run pipeline on golden set and build RAGAS EvaluationDataset.

    Returns (EvaluationDataset, per_question_data).
    """
    items = golden_set[:limit] if limit else golden_set
    samples = []
    per_question_data = []

    for item in items:
        qid = item["id"]
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_pages = item["expected_pages"]

        log.info(f"[Q{qid}] {question}")

        # Run pipeline
        pipeline_result = run_pipeline(question)
        answer = pipeline_result["answer"]
        contexts = pipeline_result["contexts"]
        chunks = pipeline_result["chunks"]

        # Page accuracy (custom metric)
        page_acc = compute_page_accuracy(expected_pages, chunks)

        # Build RAGAS sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=expected_answer,
        )
        samples.append(sample)

        per_question_data.append({
            "id": qid,
            "question": question,
            "expected_answer": expected_answer,
            "expected_pages": expected_pages,
            "generated_answer": answer,
            "retrieved_pages": sorted(set(
                int(c.get("page", 0)) for c in chunks if c.get("page") is not None
            )),
            "num_chunks": len(chunks),
            "page_accuracy": page_acc,
        })

        log.info(f"[Q{qid}] page_accuracy={page_acc:.2f}, chunks={len(chunks)}")

    dataset = EvaluationDataset(samples=samples)
    return dataset, per_question_data


def run_evaluation(limit: int = None) -> dict:
    """Run full RAGAS evaluation and return results."""
    # Load golden set
    golden_set = load_golden_set()
    log.info(f"Loaded {len(golden_set)} golden questions")

    # Build dataset by running pipeline
    dataset, per_question_data = build_eval_dataset(golden_set, limit=limit)

    # Ensure GOOGLE_API_KEY is in env for RAGAS internal Google SDK calls
    os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)

    # Configure Gemini as judge LLM and embeddings
    client = genai.Client(api_key=settings.google_api_key)
    judge_llm = llm_factory(model="gemini-2.0-flash", provider="google", client=client)
    judge_embeddings = GoogleEmbeddingsAdapter(
        embedding_factory(provider="google", model="gemini-embedding-001", client=client)
    )

    # Define metrics
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm),
        LLMContextPrecisionWithoutReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ]

    log.info("Running RAGAS evaluation...")
    ragas_result = evaluate(dataset=dataset, metrics=metrics, embeddings=judge_embeddings)

    # Extract per-question RAGAS scores
    # Map RAGAS output column names to our display names
    df = ragas_result.to_pandas()
    ragas_to_display = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "llm_context_precision_without_reference": "context_precision",
        "context_recall": "context_recall",
    }

    for i, row in df.iterrows():
        for ragas_name, display_name in ragas_to_display.items():
            if ragas_name in row:
                val = row[ragas_name]
                per_question_data[i][display_name] = float(val) if val == val else None  # handle NaN

    # Compute aggregates
    display_names = list(ragas_to_display.values())
    aggregates = {}
    all_metrics = display_names + ["page_accuracy"]
    for m in all_metrics:
        values = [q[m] for q in per_question_data if q.get(m) is not None]
        aggregates[m] = sum(values) / len(values) if values else None

    results = {
        "num_questions": len(per_question_data),
        "aggregates": aggregates,
        "per_question": per_question_data,
    }

    # Save results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved to {RESULTS_PATH}")

    return results
