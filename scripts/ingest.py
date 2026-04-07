"""
Ingestion script.
Runs the full ingestion pipeline on a folder of documents.
Loads → Chunks → Embeds → Saves to ChromaDB.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --reset
    python scripts/ingest.py --folder docs/
"""

import sys
import time
import argparse

sys.path.insert(0, ".")

from src.ingestion.loader   import load_all_documents
from src.ingestion.chunker  import chunk_documents
from src.ingestion.embedder import embed_chunks
from src.storage.vector_store import (
    save_chunks,
    reset_collection,
    get_collection_count,
)


def run_ingestion(folder: str, reset: bool = False) -> None:
    """
    Full ingestion pipeline.

    Args:
        folder: path to the documents folder
        reset:  if True, wipe ChromaDB before ingesting
    """
    start = time.time()

    print("=" * 50)
    print("  ContextAgent — Ingestion Pipeline")
    print("=" * 50)

    if reset:
        print("\n  Resetting ChromaDB...")
        reset_collection()

    already_stored = get_collection_count()
    if already_stored > 0 and not reset:
        print(f"\n  ChromaDB already contains {already_stored} chunks.")
        print("  Use --reset to wipe and re-ingest.")
        print("  Skipping ingestion.")
        return

    print(f"\n  Step 1/3 — Loading documents from '{folder}'")
    print("-" * 50)
    docs = load_all_documents(folder)

    if not docs:
        print("  No documents found. Add files to the docs/ folder.")
        return

    print(f"\n  Step 2/3 — Chunking {len(docs)} documents")
    print("-" * 50)
    chunks = chunk_documents(docs)

    print(f"\n  Step 3/3 — Embedding and saving {len(chunks)} chunks")
    print("-" * 50)
    embedded_chunks = embed_chunks(chunks)
    save_chunks(embedded_chunks)

    elapsed = round(time.time() - start, 2)

    print("\n" + "=" * 50)
    print("  Ingestion complete.")
    print(f"  Documents : {len(docs)}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Time      : {elapsed}s")
    print(f"  ChromaDB  : {get_collection_count()} chunks stored")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextAgent ingestion pipeline")
    parser.add_argument(
        "--folder",
        type=str,
        default="docs/",
        help="Path to documents folder (default: docs/)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe ChromaDB before ingesting",
    )
    args = parser.parse_args()
    run_ingestion(folder=args.folder, reset=args.reset)