"""
Vector store module.
Manages ChromaDB: saving, loading, and querying embeddings.
Persists data to disk so ingestion only runs once.
"""

import os
import uuid
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "contextagent"
TOP_K_RESULTS   = int(os.getenv("TOP_K_RESULTS", 5))


def get_collection() -> chromadb.Collection:
    """
    Connect to ChromaDB and return the collection.
    Creates the collection if it does not exist yet.
    Creates the chroma_db folder if it does not exist.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def save_chunks(chunks: list[dict]) -> None:
    """
    Save embedded chunks into ChromaDB.
    Each chunk needs a unique ID, its embedding vector,
    its original text, and its metadata.

    Args:
        chunks: list of chunk dicts with 'embedding' field
    """
    collection = get_collection()

    ids         = []
    embeddings  = []
    documents   = []
    metadatas   = []

    for chunk in chunks:
        ids.append(str(uuid.uuid4()))
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "source":      chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "file_type":   chunk["file_type"],
        })

    collection.add(
        ids        = ids,
        embeddings = embeddings,
        documents  = documents,
        metadatas  = metadatas,
    )

    print(f"  Saved {len(chunks)} chunks to ChromaDB.")
    print(f"  Collection total: {collection.count()} chunks")


def query_collection(query_embedding: list[float], top_k: int = TOP_K_RESULTS) -> list[dict]:
    """
    Search ChromaDB for the most similar chunks to a query vector.

    Args:
        query_embedding: the embedded query vector from embed_query()
        top_k: number of results to return

    Returns:
        list of result dicts:
        {
            "text":        the chunk content,
            "source":      original filename,
            "chunk_index": position in source document,
            "distance":    similarity score (lower = more similar)
        }
    """
    collection = get_collection()

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["documents"][0])):
        output.append({
            "text":        results["documents"][0][i],
            "source":      results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance":    round(results["distances"][0][i], 4),
        })

    return output


def get_collection_count() -> int:
    """Return the total number of chunks stored."""
    return get_collection().count()


def reset_collection() -> None:
    """
    Delete and recreate the collection.
    Use this when you want to re-ingest documents from scratch.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    client.delete_collection(name=COLLECTION_NAME)
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print("  Collection reset. ChromaDB is now empty.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.ingestion.loader   import load_all_documents
    from src.ingestion.chunker  import chunk_documents
    from src.ingestion.embedder import embed_chunks, embed_query

    print("Testing vector store...\n")

    docs   = load_all_documents("docs/")
    chunks = chunk_documents(docs)

    print("\n  Embedding 5 chunks for test...")
    sample = embed_chunks(chunks[:5])

    print("\n  Saving to ChromaDB...")
    save_chunks(sample)

    print("\n  Testing query...")
    query   = "What is a Python class?"
    q_vec   = embed_query(query)
    results = query_collection(q_vec, top_k=3)

    print(f"\n  Query: '{query}'")
    print(f"  Top {len(results)} results:\n")
    for r in results:
        print(f"  [{r['source']} | chunk #{r['chunk_index']} | dist={r['distance']}]")
        print(f"  {r['text'][:200]}...")
        print()