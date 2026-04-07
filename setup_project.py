import os

structure = {
    "docs": ["sample_placeholder.txt"],
    "src/ingestion": ["__init__.py", "loader.py", "chunker.py", "embedder.py"],
    "src/storage": ["__init__.py", "vector_store.py"],
    "src/retrieval": ["__init__.py", "retriever.py"],
    "src/generation": ["__init__.py", "prompt.py", "generator.py"],
    "src/memory": ["__init__.py", "buffer.py"],
    "src/agent": ["__init__.py", "orchestrator.py"],
    "scripts": ["ingest.py", "chat.py"],
    "tests": ["__init__.py", "test_retrieval.py"],
    "n8n/workflows": ["readme.txt"],
}

root_files = {
    "README.md": "# ContextAgent\n\nAgentic RAG system for querying technical documentation.\n",

    ".env.example": (
        "OPENAI_API_KEY=your_openai_key_here\n"
        "CHROMA_DB_PATH=./chroma_db\n"
        "CHUNK_SIZE=500\n"
        "CHUNK_OVERLAP=50\n"
        "TOP_K_RESULTS=5\n"
        "MEMORY_WINDOW=6\n"
    ),

    ".gitignore": (
        ".env\n"
        "__pycache__/\n"
        "*.pyc\n"
        ".venv/\n"
        "chroma_db/\n"
        "*.pdf\n"
        "*.log\n"
        ".DS_Store\n"
    ),

    "requirements.txt": (
        "openai\n"
        "langchain\n"
        "langchain-openai\n"
        "langchain-community\n"
        "chromadb\n"
        "pymupdf\n"
        "markdown\n"
        "python-dotenv\n"
        "tiktoken\n"
    ),

    "src/__init__.py": "",
}

file_stubs = {
    "src/ingestion/loader.py": (
        '"""\nLoader module.\nResponsible for reading PDF and Markdown files into raw text.\n"""\n'
    ),
    "src/ingestion/chunker.py": (
        '"""\nChunker module.\nSplits raw text into overlapping chunks using recursive character splitting.\n"""\n'
    ),
    "src/ingestion/embedder.py": (
        '"""\nEmbedder module.\nConverts text chunks into vector embeddings using OpenAI.\n"""\n'
    ),
    "src/storage/vector_store.py": (
        '"""\nVector store module.\nManages ChromaDB: saving, loading, and querying embeddings.\n"""\n'
    ),
    "src/retrieval/retriever.py": (
        '"""\nRetriever module.\nPerforms semantic similarity search against the vector store.\n"""\n'
    ),
    "src/generation/prompt.py": (
        '"""\nPrompt module.\nDefines and manages the prompt templates used for answer generation.\n"""\n'
    ),
    "src/generation/generator.py": (
        '"""\nGenerator module.\nSends the prompt + retrieved context to the LLM and returns a cited answer.\n"""\n'
    ),
    "src/memory/buffer.py": (
        '"""\nMemory buffer module.\nMaintains a sliding window of recent conversation turns.\n"""\n'
    ),
    "src/agent/orchestrator.py": (
        '"""\nOrchestrator module.\nCoordinates memory, retrieval, and generation for each user query.\n"""\n'
    ),
    "scripts/ingest.py": (
        '"""\nRun this script to ingest a document into the vector store.\nUsage: python scripts/ingest.py --file docs/yourfile.pdf\n"""\n'
    ),
    "scripts/chat.py": (
        '"""\nRun this script to start a conversation with your documents.\nUsage: python scripts/chat.py\n"""\n'
    ),
    "tests/test_retrieval.py": (
        '"""\nTests for the retrieval pipeline.\n"""\n'
    ),
    "docs/sample_placeholder.txt": (
        "Place your PDF or Markdown documents in this folder.\n"
    ),
    "n8n/workflows/readme.txt": (
        "Export n8n workflow JSON files into this folder.\n"
    ),
}

def create_structure():
    base = os.getcwd()
    print(f"Creating ContextAgent project structure in: {base}\n")

    for folder in structure:
        os.makedirs(folder, exist_ok=True)
        print(f"  created folder: {folder}/")

    for path, content in root_files.items():
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        print(f"  created file:   {path}")

    for path, content in file_stubs.items():
        with open(path, "w") as f:
            f.write(content)
        print(f"  created file:   {path}")

    print("\nDone. Your project structure is ready.")
    print("Next: create a virtual environment and install dependencies.")
    print("  python -m venv .venv")
    print("  source .venv/bin/activate   # Windows: .venv\\Scripts\\activate")
    print("  pip install -r requirements.txt")

if __name__ == "__main__":
    create_structure()