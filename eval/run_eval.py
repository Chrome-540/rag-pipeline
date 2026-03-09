"""CLI runner for RAGAS evaluation."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.evaluator import run_evaluation


def print_results(results: dict):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 70)

    # Per-question scores
    print(f"\n{'ID':<4} {'Question':<45} {'Faith':>6} {'Relev':>6} {'Prec':>6} {'Recall':>6} {'Page':>6}")
    print("-" * 82)

    for q in results["per_question"]:
        question = q["question"][:43] + ".." if len(q["question"]) > 45 else q["question"]
        faith = f"{q.get('faithfulness', 0):.2f}" if q.get("faithfulness") is not None else "  N/A"
        relev = f"{q.get('answer_relevancy', 0):.2f}" if q.get("answer_relevancy") is not None else "  N/A"
        prec = f"{q.get('context_precision', 0):.2f}" if q.get("context_precision") is not None else "  N/A"
        recall = f"{q.get('context_recall', 0):.2f}" if q.get("context_recall") is not None else "  N/A"
        page = f"{q['page_accuracy']:.2f}"

        print(f"Q{q['id']:<3} {question:<45} {faith:>6} {relev:>6} {prec:>6} {recall:>6} {page:>6}")

    # Aggregates
    agg = results["aggregates"]
    print("-" * 82)
    print(f"{'AVG':<49} ", end="")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "page_accuracy"]:
        val = agg.get(m)
        print(f"{val:>6.2f} " if val is not None else "  N/A  ", end="")
    print()
    print("=" * 70)

    print(f"\nResults saved to eval/results.json")


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on golden set")
    parser.add_argument("--test", type=int, default=None, help="Run only first N questions")
    args = parser.parse_args()

    limit = args.test
    count = limit or 20
    print(f"Running evaluation on {count} question(s)...")

    results = run_evaluation(limit=limit)
    print_results(results)


if __name__ == "__main__":
    main()
