"""
Retriever module — v3.
Hybrid search: BM25 + vector + RRF fusion + cross-encoder reranking.

Pipeline:
  1. Vector search (semantic)  — top 20 candidates
  2. BM25 search (keyword)     — top 20 candidates
  3. RRF fusion                — merge into top 20
  4. Cross-encoder reranking   — rerank top 20, return top 5
"""

import os
import sys
import math
from collections import defaultdict

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable
from src.ingestion.embedder   import embed_query
from src.storage.vector_store import query_collection

TOP_K        = int(os.getenv("TOP_K_RESULTS", 5))
MAX_DISTANCE = 0.85
RRF_K        = 60
RERANK_TOP_N = 20

# ── Cross-encoder lazy loader ───────────────────────────────────
_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print("  Loading cross-encoder model (first time only)...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("  Cross-encoder ready.")
    return _cross_encoder

# ── BM25 in-memory index ────────────────────────────────────────
_bm25_corpus: list[dict] = []
_bm25_ready:  bool       = False


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _build_bm25_index() -> None:
    global _bm25_corpus, _bm25_ready
    from src.storage.vector_store import get_collection

    collection = get_collection()
    results    = collection.get(include=["documents", "metadatas"])

    _bm25_corpus = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        _bm25_corpus.append({
            "text":        doc,
            "source":      meta.get("source", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "file_type":   meta.get("file_type", ""),
            "tokens":      _tokenize(doc),
        })
    _bm25_ready = True


def _bm25_search(query: str, top_k: int) -> list[dict]:
    global _bm25_corpus, _bm25_ready

    if not _bm25_ready:
        _build_bm25_index()

    if not _bm25_corpus:
        return []

    query_tokens = _tokenize(query)
    N            = len(_bm25_corpus)
    k1, b        = 1.5, 0.75
    avg_dl       = sum(len(c["tokens"]) for c in _bm25_corpus) / N

    df: dict[str, int] = defaultdict(int)
    for chunk in _bm25_corpus:
        for term in set(chunk["tokens"]):
            df[term] += 1

    scores = []
    for chunk in _bm25_corpus:
        score  = 0.0
        dl     = len(chunk["tokens"])
        tf_map = defaultdict(int)
        for t in chunk["tokens"]:
            tf_map[t] += 1

        for term in query_tokens:
            if term not in tf_map:
                continue
            tf      = tf_map[term]
            idf     = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score  += idf * tf_norm

        scores.append((score, chunk))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "text":        c["text"],
            "source":      c["source"],
            "chunk_index": c["chunk_index"],
            "file_type":   c["file_type"],
            "bm25_score":  s,
            "distance":    0.5,
        }
        for s, c in scores[:top_k]
    ]


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results:   list[dict],
    top_k:          int,
) -> list[dict]:
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map:  dict[str, dict]  = {}

    for rank, chunk in enumerate(vector_results):
        key = f"{chunk['source']}_{chunk['chunk_index']}"
        rrf_scores[key] += 1.0 / (RRF_K + rank + 1)
        chunk_map[key]   = chunk

    for rank, chunk in enumerate(bm25_results):
        key = f"{chunk['source']}_{chunk['chunk_index']}"
        rrf_scores[key] += 0.5 / (RRF_K + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = chunk

    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    return [chunk_map[k] for k in sorted_keys[:top_k]]


def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Cross-encoder reranking.
    Reads query + chunk together for precise relevance scoring.
    """
    encoder = _get_cross_encoder()
    pairs   = [[query, c["text"]] for c in candidates]
    scores  = encoder.predict(pairs)

    for i, chunk in enumerate(candidates):
        chunk["rerank_score"] = float(scores[i])

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


@traceable(name="retrieve")
def retrieve(query: str, top_k: int = TOP_K, original_query: str = None) -> list[dict]:
    """
    Full retrieval pipeline:
      1. Vector search on query
      2. BM25 on original_query
      3. RRF fusion → top RERANK_TOP_N candidates
      4. Cross-encoder reranking → top top_k results
    """
    bm25_query = original_query if original_query else query

    # Stage 1 — vector search
    query_vector   = embed_query(query)
    vector_results = query_collection(
        query_embedding = query_vector,
        top_k           = RERANK_TOP_N,
    )
    filtered = [r for r in vector_results if r["distance"] <= MAX_DISTANCE]

    # Stage 2 — BM25
    bm25_results = _bm25_search(bm25_query, top_k=RERANK_TOP_N)

    # Stage 3 — RRF fusion
    candidates = _reciprocal_rank_fusion(filtered, bm25_results, top_k=RERANK_TOP_N)

    if not candidates:
        print("  Warning: no relevant chunks found.")
        return []

    # Stage 4 — cross-encoder reranking
    reranked = _rerank(query, candidates, top_k=top_k)

    return reranked


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    questions = [
        "What is a Python class?",
        "How does error handling work in Python?",
        "What is retrieval augmented generation?",
        "What is the attention mechanism?",
    ]

    print("Testing retriever v3 with cross-encoder reranking...\n")

    for query in questions:
        print(f"  Query: '{query}'")
        results = retrieve(query, top_k=5, original_query=query)
        for r in results:
            score = f"rerank={r.get('rerank_score', 0):.3f}"
            print(f"    [{r['source']} | dist={r['distance']:.3f} | {score}]")
        print()