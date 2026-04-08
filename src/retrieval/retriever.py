"""
Retriever module — v2.
Hybrid search: combines vector similarity (semantic) with BM25 (keyword).
Results are fused with Reciprocal Rank Fusion (RRF).

Why hybrid?
- Vector search finds semantically similar chunks (meaning-based)
- BM25 finds exact keyword matches (term-based)
- RRF combines both rankings — best of both worlds
"""

import os
import sys
import math
from collections import defaultdict
sys.path.insert(0, ".")

from src.ingestion.embedder   import embed_query
from src.storage.vector_store import query_collection
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

TOP_K        = int(os.getenv("TOP_K_RESULTS", 5))
MAX_DISTANCE = 0.85
RRF_K        = 60  # RRF constant — standard value, higher = smoother fusion

# ── BM25 in-memory index ────────────────────────────────────────
_bm25_corpus: list[dict] = []
_bm25_ready:  bool       = False


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def _build_bm25_index() -> None:
    """
    Load all chunks from ChromaDB into memory for BM25 search.
    Called once on first retrieval — lazy initialization.
    """
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
    """
    BM25 keyword search over the in-memory corpus.
    Returns top_k results ranked by BM25 score.
    """
    global _bm25_corpus, _bm25_ready

    if not _bm25_ready:
        _build_bm25_index()

    if not _bm25_corpus:
        return []

    query_tokens = _tokenize(query)
    N            = len(_bm25_corpus)
    k1, b        = 1.5, 0.75

    # Compute average document length
    avg_dl = sum(len(c["tokens"]) for c in _bm25_corpus) / N

    # Document frequency per term
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
            tf  = tf_map[term]
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
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
            "distance":    0.5,  # placeholder — BM25 has no cosine distance
        }
        for s, c in scores[:top_k]
    ]


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results:   list[dict],
    top_k:          int,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    score(d) = Σ 1/(k + rank(d))
    Higher score = better combined rank.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map:  dict[str, dict]  = {}

    for rank, chunk in enumerate(vector_results):
        key = f"{chunk['source']}_{chunk['chunk_index']}"
        rrf_scores[key] += 1.0 / (RRF_K + rank + 1)
        chunk_map[key]   = chunk

    for rank, chunk in enumerate(bm25_results):
        key = f"{chunk['source']}_{chunk['chunk_index']}"
        rrf_scores[key] += 1.0 / (RRF_K + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = chunk

    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    return [chunk_map[k] for k in sorted_keys[:top_k]]


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Hybrid retrieval: vector + BM25 fused with RRF.

    Steps:
        1. Embed query → vector search (semantic)
        2. Tokenize query → BM25 search (keyword)
        3. Fuse both ranked lists with RRF
        4. Filter by MAX_DISTANCE on vector results
        5. Return top_k fused results
    """
    # Vector search
    query_vector   = embed_query(query)
    vector_results = query_collection(query_embedding=query_vector, top_k=top_k * 2)
    filtered       = [r for r in vector_results if r["distance"] <= MAX_DISTANCE]

    # BM25 search
    bm25_results   = _bm25_search(query, top_k=top_k * 2)

    # Fuse
    fused = _reciprocal_rank_fusion(filtered, bm25_results, top_k=top_k)

    if not fused:
        print(f"  Warning: no relevant chunks found for query.")

    return fused


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a labelled context string."""
    if not chunks:
        return "No relevant context found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)