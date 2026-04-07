"""
Embedder module.
Converts text chunks into vector embeddings using OpenAI.
Sends chunks in batches to avoid API rate limits.
Attaches the embedding vector to each chunk dict.
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE      = 100  # number of chunks per API call


def get_client() -> OpenAI:
    """
    Create and return an OpenAI client.
    Raises a clear error if the API key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Make sure it is set in your .env file."
        )
    return OpenAI(api_key=api_key)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' field to every chunk dict.
    Sends chunks to OpenAI in batches of BATCH_SIZE.

    Args:
        chunks: list of chunk dicts from chunker.py

    Returns:
        same list with 'embedding' added to each dict
    """
    client      = get_client()
    total       = len(chunks)
    embedded    = 0

    print(f"  Embedding {total} chunks in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch      = chunks[i : i + BATCH_SIZE]
        texts      = [chunk["text"] for chunk in batch]

        response   = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        for j, embedding_obj in enumerate(response.data):
            batch[j]["embedding"] = embedding_obj.embedding

        embedded += len(batch)
        print(f"  Progress: {embedded}/{total} chunks embedded")

        if i + BATCH_SIZE < total:
            time.sleep(0.5)

    print(f"\n  Done. All {total} chunks embedded.")
    print(f"  Embedding dimension: {len(chunks[0]['embedding'])}")
    return chunks


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string at retrieval time.
    Returns the raw vector (list of floats).

    Args:
        query: the user's question

    Returns:
        list of floats — the embedding vector
    """
    client   = get_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.ingestion.loader  import load_all_documents
    from src.ingestion.chunker import chunk_documents

    print("Testing embedder...\n")

    docs   = load_all_documents("docs/")
    chunks = chunk_documents(docs)

    first_three = chunks[:3]
    embedded    = embed_chunks(first_three)

    print("\n--- Sample result ---\n")
    for chunk in embedded:
        print(f"  Source    : {chunk['source']}")
        print(f"  Chunk #   : {chunk['chunk_index']}")
        print(f"  Embedding : [{chunk['embedding'][0]:.6f}, "
              f"{chunk['embedding'][1]:.6f}, ... ] "
              f"({len(chunk['embedding'])} dims)")
        print()