"""
Chunker module.
Splits raw document text into overlapping chunks.
Uses LangChain's RecursiveCharacterTextSplitter.
Each chunk carries its source metadata for citation later.
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def chunk_document(document: dict) -> list[dict]:
    """
    Split a single document dict (from loader) into chunks.

    Args:
        document: dict with keys "text", "source", "file_type", "num_pages"

    Returns:
        list of chunk dicts:
        {
            "text":        the chunk content,
            "source":      original filename,
            "file_type":   .txt / .md / .pdf,
            "chunk_index": position of this chunk in the document
        }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(document["text"])

    chunks = []
    for i, text in enumerate(raw_chunks):
        if text.strip():
            chunks.append({
                "text":        text.strip(),
                "source":      document["source"],
                "file_type":   document["file_type"],
                "chunk_index": i,
            })

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk every document in a list.

    Args:
        documents: list of document dicts from load_all_documents()

    Returns:
        flat list of all chunks across all documents
    """
    all_chunks = []

    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']} → {len(chunks)} chunks")

    print(f"\n  Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.ingestion.loader import load_all_documents

    print("Testing chunker...\n")
    print(f"  Settings: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}\n")

    docs   = load_all_documents("docs/")
    chunks = chunk_documents(docs)

    print("\n--- Sample chunks ---\n")
    for chunk in chunks[:3]:
        print(f"  Source : {chunk['source']} (chunk #{chunk['chunk_index']})")
        print(f"  Length : {len(chunk['text'])} chars")
        print(f"  Text   : {chunk['text'][:200]}...")
        print()