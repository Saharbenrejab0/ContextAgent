"""
ContextAgent — Deterministic Evaluation
No LLM judge — reproducible metrics every run.

Metrics:
  keyword_coverage  : does the answer contain expected keywords?
  source_accuracy   : is the right document cited?
  retrieval_score   : avg cosine distance of retrieved chunks (lower = better)
  answer_completeness: does the answer avoid "I could not find"?
  chunk_coverage    : were enough chunks retrieved?

Usage:
    python scripts/evaluate_ragas.py
    python scripts/evaluate_ragas.py --output results/eval_baseline.json
"""

import sys, os, json, argparse
from datetime import datetime

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.retriever  import retrieve, format_context
from src.generation.prompt    import build_messages
from src.generation.generator import generate

# ── Test dataset ─────────────────────────────────────────────────────────────
# Each entry has:
#   question      : the query
#   keywords      : words that MUST appear in a correct answer
#   expected_source: which document should be cited
TEST_DATASET = [
    {
        "question":        "What is a Python class?",
        "keywords":        ["class", "object", "blueprint", "OOP", "inheritance"],
        "expected_source": "classes.txt",
    },
    {
        "question":        "How do you define a class in Python?",
        "keywords":        ["class", "keyword", "def", "ClassName"],
        "expected_source": "classes.txt",
    },
    {
        "question":        "What is inheritance in Python?",
        "keywords":        ["inherit", "base", "derived", "subclass", "parent"],
        "expected_source": "classes.txt",
    },
    {
        "question":        "How does error handling work in Python?",
        "keywords":        ["try", "except", "exception", "error", "raise"],
        "expected_source": "errors.txt",
    },
    {
        "question":        "What is the difference between a syntax error and an exception?",
        "keywords":        ["syntax", "exception", "runtime", "parsing"],
        "expected_source": "errors.txt",
    },
    {
        "question":        "How do Python modules work?",
        "keywords":        ["module", "import", "file", ".py"],
        "expected_source": "modules.txt",
    },
    {
        "question":        "What is a Python package?",
        "keywords":        ["package", "directory", "module", "__init__"],
        "expected_source": "modules.txt",
    },
    {
        "question":        "What are Python data structures?",
        "keywords":        ["list", "dict", "tuple", "set"],
        "expected_source": "datastructures.txt",
    },
    {
        "question":        "What is retrieval augmented generation?",
        "keywords":        ["retrieval", "generation", "RAG", "knowledge", "parametric"],
        "expected_source": "rag_paper.pdf",
    },
    {
        "question":        "What problem does RAG solve?",
        "keywords":        ["hallucination", "knowledge", "retrieval", "parametric", "external"],
        "expected_source": "rag_paper.pdf",
    },
    {
        "question":        "What is the attention mechanism?",
        "keywords":        ["attention", "query", "key", "value", "transformer"],
        "expected_source": "attention_paper.pdf",
    },
    {
        "question":        "What does the __init__ method do in Python?",
        "keywords":        ["__init__", "constructor", "instance", "initialize"],
        "expected_source": "classes.txt",
    },
    {
        "question":        "How do you import a module in Python?",
        "keywords":        ["import", "module", "from", "namespace"],
        "expected_source": "modules.txt",
    },
    {
        "question":        "What is a list comprehension in Python?",
        "keywords":        ["comprehension", "list", "expression", "for", "filter"],
        "expected_source": "datastructures.txt",
    },
    {
        "question":        "What is the for loop used for in Python?",
        "keywords":        ["for", "iterate", "sequence", "loop"],
        "expected_source": "controlflow.txt",
    },
]


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """
    Fraction of expected keywords found in the answer.
    Case-insensitive. Returns 0.0 to 1.0.
    """
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(found / len(keywords), 4)


def source_accuracy(sources: list[str], expected: str) -> float:
    """
    1.0 if the expected source is cited, 0.0 otherwise.
    """
    return 1.0 if any(expected in s for s in sources) else 0.0


def retrieval_score(chunks: list[dict]) -> float:
    """
    Converts avg distance to a 0-1 score where 1.0 = perfect retrieval.
    retrieval_score = 1 - avg_distance
    """
    if not chunks:
        return 0.0
    avg_dist = sum(c["distance"] for c in chunks) / len(chunks)
    return round(1.0 - avg_dist, 4)


def answer_completeness(answer: str) -> float:
    """
    1.0 if the system gave a real answer.
    0.0 if it said "I could not find".
    """
    if "could not find" in answer.lower():
        return 0.0
    if len(answer.strip()) < 30:
        return 0.0
    return 1.0


def chunk_coverage(chunks: list[dict], top_k: int = 5) -> float:
    """
    Fraction of expected chunks retrieved (at least top_k/2 chunks found).
    """
    if not chunks:
        return 0.0
    return round(min(len(chunks) / top_k, 1.0), 4)


def evaluate_one(item: dict, top_k: int = 5) -> dict:
    """Run the full pipeline and score one question deterministically."""
    q       = item["question"]
    chunks  = retrieve(q, top_k=top_k)
    context = format_context(chunks)
    msgs    = build_messages(question=q, context=context)
    result  = generate(msgs)
    answer  = result["answer"]
    sources = result["sources"]

    kw_score  = keyword_coverage(answer, item["keywords"])
    src_score = source_accuracy(sources, item["expected_source"])
    ret_score = retrieval_score(chunks)
    comp      = answer_completeness(answer)
    chk       = chunk_coverage(chunks, top_k)

    composite = round((kw_score + src_score + ret_score + comp + chk) / 5, 4)

    return {
        "question":           q,
        "keyword_coverage":   kw_score,
        "source_accuracy":    src_score,
        "retrieval_score":    ret_score,
        "answer_completeness":comp,
        "chunk_coverage":     chk,
        "composite":          composite,
        "answer_snippet":     answer[:120],
        "sources_cited":      sources,
        "chunks_found":       len(chunks),
        "avg_distance":       round(sum(c["distance"] for c in chunks)/len(chunks), 4) if chunks else 1.0,
        "tokens":             result["tokens"]["total"],
    }


def run_evaluation(output_path: str = None) -> dict:
    print("\n" + "="*57)
    print("  ContextAgent — Deterministic Evaluation")
    print("="*57)
    print(f"\n  {len(TEST_DATASET)} questions — fully reproducible\n")

    per_question = []
    total_tokens = 0

    for i, item in enumerate(TEST_DATASET):
        q = item["question"]
        print(f"  [{i+1:02d}/{len(TEST_DATASET)}] {q[:52]}...")
        r = evaluate_one(item)
        per_question.append(r)
        total_tokens += r["tokens"]
        print(f"         kw={r['keyword_coverage']:.2f}  "
              f"src={r['source_accuracy']:.2f}  "
              f"ret={r['retrieval_score']:.2f}  "
              f"comp={r['answer_completeness']:.2f}  "
              f"→ {r['composite']:.2f}")

    def avg(key):
        return round(sum(r[key] for r in per_question) / len(per_question), 4)

    results = {
        "timestamp":          datetime.now().isoformat(),
        "method":             "deterministic — no LLM judge",
        "num_questions":      len(TEST_DATASET),
        "total_tokens":       total_tokens,
        "keyword_coverage":   avg("keyword_coverage"),
        "source_accuracy":    avg("source_accuracy"),
        "retrieval_score":    avg("retrieval_score"),
        "answer_completeness":avg("answer_completeness"),
        "chunk_coverage":     avg("chunk_coverage"),
        "avg_score":          avg("composite"),
        "per_question":       per_question,
    }

    print("\n" + "="*57)
    print("  Results")
    print("="*57)
    print(f"  Keyword coverage   : {results['keyword_coverage']:.4f}  "
          f"{'✅' if results['keyword_coverage']   > 0.70 else '⚠️'}")
    print(f"  Source accuracy    : {results['source_accuracy']:.4f}  "
          f"{'✅' if results['source_accuracy']    > 0.80 else '⚠️'}")
    print(f"  Retrieval score    : {results['retrieval_score']:.4f}  "
          f"{'✅' if results['retrieval_score']    > 0.50 else '⚠️'}")
    print(f"  Answer completeness: {results['answer_completeness']:.4f}  "
          f"{'✅' if results['answer_completeness']> 0.90 else '⚠️'}")
    print(f"  Chunk coverage     : {results['chunk_coverage']:.4f}  "
          f"{'✅' if results['chunk_coverage']     > 0.80 else '⚠️'}")
    print(f"\n  Composite score    : {results['avg_score']:.4f}")
    print(f"  Total tokens used  : {results['total_tokens']:,}")
    print("="*57)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved → {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default="results/eval_deterministic.json")
    args = parser.parse_args()
    run_evaluation(output_path=args.output)