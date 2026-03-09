"""Test generation pipeline."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generation import generate_answer


def test_generate():
    queries = [
        "Why is Venus the hottest planet?",
        "What is the greenhouse effect?",
        "What are the planets in our solar system?",
    ]

    for query in queries:
        print(f"Q: {query}")
        result = generate_answer(query)
        print(f"A: {result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  {s['source']} - Page {s['page']}")
        print("-" * 50)


if __name__ == "__main__":
    test_generate()
