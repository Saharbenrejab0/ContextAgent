"""
Retriever module.
Takes a user question, embeds it, and fetches the most
relevant chunks from ChromaDB using cosine similarity.
"""

import os
import sys
sys.path.insert(0, ".")

from src.ingestion.embedder   import embed_query
from src.storage.vector_store import query_collection
from dotenv import load_dotenv

load_dotenv()

TOP_K = int(os.getenv("TOP_K_RESULTS", 5))
MAX_DISTANCE = 0.85


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve the most relevant chunks for a given query.

    Steps:
        1. Embed the query string into a vector
        2. Search ChromaDB for the closest chunk vectors
        3. Filter out chunks that are too distant (low relevance)
        4. Return the results

    Args:
        query: the user's question as plain text
        top_k: how many chunks to retrieve

    Returns:
        list of chunk dicts:
        {
            "text":        chunk content,
            "source":      filename,
            "chunk_index": position in source doc,
            "distance":    cosine distance (0=identical, 1=unrelated)
        }
    """
    query_vector = embed_query(query)

    results = query_collection(
        query_embedding=query_vector,
        top_k=top_k,
    )

    filtered = [r for r in results if r["distance"] <= MAX_DISTANCE]

    if not filtered:
        print(f"  Warning: no relevant chunks found for query (all distances > {MAX_DISTANCE})")

    return filtered


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string
    ready to be injected into the LLM prompt.

    Each chunk is labelled with its source file so the
    generator can produce citations.

    Args:
        chunks: list of chunk dicts from retrieve()

    Returns:
        formatted string with all chunks and their sources
    """
    if not chunks:
        return "No relevant context found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    queries = [
        "What is a Python class?",
        "How does error handling work in Python?",
        "What is retrieval augmented generation?",
        "What is the attention mechanism?",
    ]

    print("Testing retriever...\n")

    for query in queries:
        print(f"  Query: '{query}'")
        results = retrieve(query, top_k=3)

        if results:
            print(f"  Found {len(results)} relevant chunks:\n")
            for r in results:
                print(f"    [{r['source']} | chunk #{r['chunk_index']} | dist={r['distance']}]")
                print(f"    {r['text'][:150]}...")
                print()
        else:
            print("  No relevant results.\n")

        print("-" * 50 + "\n")