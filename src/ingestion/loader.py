"""
Loader module.
Responsible for reading PDF, Markdown, and TXT files into raw text.
Returns a dict with the text content and basic metadata.
"""

import os
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def load_document(file_path: str) -> dict:
    """
    Load a document from disk and return its content + metadata.

    Args:
        file_path: absolute or relative path to the file

    Returns:
        {
            "text": full text content of the document,
            "source": filename,
            "file_type": extension,
            "num_pages": number of pages (PDF only, else 1)
        }

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the file extension is not supported
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    if ext == ".pdf":
        return _load_pdf(path)
    else:
        return _load_text(path, ext)


def _load_pdf(path: Path) -> dict:
    """
    Extract text from a PDF file using PyMuPDF.
    Iterates over every page and joins the text.
    """
    doc = fitz.open(str(path))
    pages_text = []

    for page in doc:
        text = page.get_text()
        if text.strip():
            pages_text.append(text)

    doc.close()

    full_text = "\n".join(pages_text)

    return {
        "text": full_text,
        "source": path.name,
        "file_type": ".pdf",
        "num_pages": len(pages_text),
    }


def _load_text(path: Path, ext: str) -> dict:
    """
    Read a plain text or Markdown file.
    Uses UTF-8 encoding with fallback to latin-1.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    return {
        "text": text,
        "source": path.name,
        "file_type": ext,
        "num_pages": 1,
    }


def load_all_documents(folder_path: str) -> list[dict]:
    """
    Load every supported document in a folder.
    Skips unsupported files silently.

    Args:
        folder_path: path to the folder (e.g. "docs/")

    Returns:
        list of document dicts, one per file
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    documents = []

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in SUPPORTED_EXTENSIONS:
            print(f"  Loading: {file.name}")
            doc = load_document(str(file))
            documents.append(doc)

    print(f"\n  Total documents loaded: {len(documents)}")
    return documents


if __name__ == "__main__":
    print("Testing loader...\n")

    docs = load_all_documents("docs/")

    for doc in docs:
        preview = doc["text"][:150].replace("\n", " ")
        print(f"  [{doc['file_type']}] {doc['source']}")
        print(f"  Pages : {doc['num_pages']}")
        print(f"  Preview: {preview}...")
        print()