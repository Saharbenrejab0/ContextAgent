"""
Evaluation Script — LLM-as-Judge
Mesure la qualité du pipeline RAG sur 15 questions test.
Pas de dépendance externe autre qu'OpenAI.

Usage:
    python scripts/evaluate_ragas.py
"""

import sys
import json
import os
import argparse
from datetime import datetime

sys.path.insert(0, ".")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from src.retrieval.retriever  import retrieve, format_context
from src.generation.prompt    import build_messages
from src.generation.generator import generate

# ── Dataset de test ─────────────────────────────────────────────
TEST_DATASET = [
    {
        "question":     "What is a Python class?",
        "ground_truth": "A Python class is a blueprint for creating objects that bundles data and functionality together. It is defined using the class keyword."
    },
    {
        "question":     "How do you define a class in Python?",
        "ground_truth": "A class in Python is defined using the class keyword followed by the class name and a colon. The body contains methods and attributes."
    },
    {
        "question":     "What is inheritance in Python?",
        "ground_truth": "Inheritance allows a class to inherit attributes and methods from another class. Python supports multiple inheritance."
    },
    {
        "question":     "How does error handling work in Python?",
        "ground_truth": "Python uses try and except blocks for error handling. Code that might raise an error is in the try block, caught by the except block."
    },
    {
        "question":     "What is the difference between a syntax error and an exception?",
        "ground_truth": "A syntax error occurs when code is not written correctly. An exception occurs during execution even if the syntax is correct."
    },
    {
        "question":     "How do Python modules work?",
        "ground_truth": "A Python module is a file containing Python definitions. You can import it using the import statement to reuse its definitions."
    },
    {
        "question":     "What is a Python package?",
        "ground_truth": "A package is a directory containing multiple Python modules and a special __init__.py file."
    },
    {
        "question":     "What are Python data structures?",
        "ground_truth": "Python provides built-in data structures including lists, tuples, sets, and dictionaries."
    },
    {
        "question":     "What is retrieval augmented generation?",
        "ground_truth": "RAG combines a pre-trained language model with a retrieval component that fetches relevant documents to ground the generation."
    },
    {
        "question":     "What problem does RAG solve?",
        "ground_truth": "RAG solves the problem of models relying solely on training memory. It allows access to external knowledge and reduces hallucinations."
    },
    {
        "question":     "What is the attention mechanism?",
        "ground_truth": "The attention mechanism allows models to focus on relevant parts of the input when generating each output token."
    },
    {
        "question":     "What does the __init__ method do in Python?",
        "ground_truth": "The __init__ method is the constructor of a Python class, called automatically when a new object is created."
    },
    {
        "question":     "How do you import a module in Python?",
        "ground_truth": "You import a module using the import statement, or from module import name for specific names."
    },
    {
        "question":     "What is a list comprehension in Python?",
        "ground_truth": "A list comprehension provides a concise way to create lists using an expression followed by a for clause in square brackets."
    },
    {
        "question":     "What is the for loop used for in Python?",
        "ground_truth": "The for loop iterates over a sequence such as a list, tuple, or string, executing a block of code for each element."
    },
]


def run_pipeline(question: str, top_k: int = 5) -> dict:
    chunks   = retrieve(question, top_k=top_k)
    context  = format_context(chunks)
    messages = build_messages(question=question, context=context)
    result   = generate(messages)
    return {
        "answer":   result["answer"],
        "contexts": [c["text"] for c in chunks],
        "tokens":   result["tokens"]["total"],
    }


def judge_with_llm(client: OpenAI, question: str, answer: str,
                   context: str, ground_truth: str) -> dict:
    prompt = f"""You are an expert evaluator for RAG systems.
Score the following on 4 metrics from 0.0 to 1.0.
Return ONLY valid JSON, no explanation, no markdown.

Question: {question}

Context provided:
{context[:1500]}

Answer given:
{answer}

Expected answer:
{ground_truth}

Metrics to score:
- faithfulness: is the answer grounded in the context only?
- answer_relevancy: does the answer address the question?
- context_precision: is the context relevant to the question?
- context_recall: does the context contain all needed information?

Return exactly:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}}"""

    response = client.chat.completions.create(
        model       = "gpt-4o-mini",
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = 100,
        temperature = 0.0,
    )

    try:
        raw    = response.choices[0].message.content.strip()
        scores = json.loads(raw)
        return {k: float(v) for k, v in scores.items()}
    except Exception:
        return {"faithfulness": 0.5, "answer_relevancy": 0.5,
                "context_precision": 0.5, "context_recall": 0.5}


def run_evaluation(output_path: str = None) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)

    print("\n" + "=" * 55)
    print("  ContextAgent — Evaluation (LLM-as-Judge)")
    print("=" * 55)
    print(f"\n  {len(TEST_DATASET)} questions — this takes ~5 minutes\n")

    all_scores   = []
    total_tokens = 0
    per_question = []

    for i, item in enumerate(TEST_DATASET):
        q = item["question"]
        print(f"  [{i+1}/{len(TEST_DATASET)}] {q[:55]}...")

        result        = run_pipeline(q, top_k=5)
        total_tokens += result["tokens"]
        context_str   = "\n\n".join(result["contexts"])

        scores = judge_with_llm(
            client       = client,
            question     = q,
            answer       = result["answer"],
            context      = context_str,
            ground_truth = item["ground_truth"],
        )

        all_scores.append(scores)
        per_question.append({
            "question":          q,
            "answer":            result["answer"][:200],
            "faithfulness":      scores["faithfulness"],
            "answer_relevancy":  scores["answer_relevancy"],
            "context_precision": scores["context_precision"],
            "context_recall":    scores["context_recall"],
        })

        print(f"         faith={scores['faithfulness']:.2f}  "
              f"rel={scores['answer_relevancy']:.2f}  "
              f"prec={scores['context_precision']:.2f}  "
              f"rec={scores['context_recall']:.2f}")

    def avg(key):
        return round(sum(s[key] for s in all_scores) / len(all_scores), 4)

    results = {
        "timestamp":         datetime.now().isoformat(),
        "method":            "LLM-as-judge (gpt-4o-mini)",
        "num_questions":     len(TEST_DATASET),
        "total_tokens":      total_tokens,
        "faithfulness":      avg("faithfulness"),
        "answer_relevancy":  avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall":    avg("context_recall"),
        "avg_score":         round((
            avg("faithfulness") + avg("answer_relevancy") +
            avg("context_precision") + avg("context_recall")
        ) / 4, 4),
        "per_question": per_question,
    }

    print("\n" + "=" * 55)
    print("  Results — Baseline v1")
    print("=" * 55)
    print(f"  Faithfulness      : {results['faithfulness']:.4f}  "
          f"{'✅' if results['faithfulness']      > 0.85 else '⚠️'}")
    print(f"  Answer Relevancy  : {results['answer_relevancy']:.4f}  "
          f"{'✅' if results['answer_relevancy']  > 0.80 else '⚠️'}")
    print(f"  Context Precision : {results['context_precision']:.4f}  "
          f"{'✅' if results['context_precision'] > 0.70 else '⚠️'}")
    print(f"  Context Recall    : {results['context_recall']:.4f}  "
          f"{'✅' if results['context_recall']    > 0.75 else '⚠️'}")
    print(f"\n  Average score     : {results['avg_score']:.4f}")
    print(f"  Total tokens used : {results['total_tokens']:,}")
    print("=" * 55)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved → {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ragas_baseline.json")
    args = parser.parse_args()
    run_evaluation(output_path=args.output)