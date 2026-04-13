# ContextAgent

A production-grade RAG (Retrieval-Augmented Generation) system that lets you ask natural language questions over your own documents and receive precise, cited answers.

Built with FastAPI, React, ChromaDB, OpenAI, and a hybrid BM25 + vector search pipeline.

---

## What it does

- Ingests PDF, TXT, and Markdown documents
- Chunks and embeds them using OpenAI text-embedding-3-small
- Stores vectors in a local ChromaDB database
- Answers questions using hybrid search (BM25 + cosine similarity + RRF fusion)
- Cites the exact source document for every answer
- Maintains conversation memory across turns
- Streams responses token by token
- Tracks every LLM call via LangSmith

---

## Interface

| Page | Description |
|---|---|
| Chat | Ask questions, see retrieved chunks in debug panel |
| Dashboard | Token usage, retrieval distances, system health |
| Documents | Upload new documents, view ingested files |
| Evaluation | Run deterministic quality evaluation |

---

## Tech stack

| Layer | Technology |
|---|---|
| Document loading | PyMuPDF, pathlib |
| Chunking | LangChain RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter |
| Embeddings | OpenAI text-embedding-3-small (1536 dims) |
| Vector store | ChromaDB (local persistent) |
| Retrieval | Hybrid BM25 + cosine vector + RRF fusion |
| LLM | GPT-4o-mini (temperature 0.1, streaming) |
| Memory | Sliding window buffer (6 turns) |
| Backend | FastAPI + uvicorn |
| Frontend | React 18 + Vite + Recharts |
| Tracing | LangSmith (@traceable) |
| Evaluation | Deterministic — 5 objective metrics, composite score 0.8720 |

---

## Setup

### Requirements

- Python 3.10+
- Node.js 18+
- OpenAI API key
- LangSmith API key (optional, for tracing)

### 1. Clone the repository

```bash
git clone https://github.com/Saharbenrejab0/ContextAgent.git
cd ContextAgent
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...your key here...
LANGCHAIN_API_KEY=lsv2__...your langsmith key... (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=contextagent
```

### 5. Add your documents

Place your PDF, TXT, or Markdown files in the `docs/` folder.

### 6. Run ingestion

```bash
python scripts/ingest.py
```

This loads, chunks, embeds, and stores all documents in ChromaDB. Run once per document set.

### 7. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 8. Windows only — patch orjson if blocked by security policy

If you see `ImportError: DLL load failed while importing orjson`, run:

```bash
python -c "
import site, os
for p in site.getsitepackages():
    f = os.path.join(p, 'chromadb', 'api', 'base_http_client.py')
    if os.path.exists(f):
        content = open(f).read()
        content = content.replace('import orjson as json', 'import json')
        open(f, 'w').write(content)
        print('Patched:', f)
"
```

---

## Running the project

Open two terminals:

**Terminal 1 — Backend**
```bash
uvicorn backend.main:app --reload
```

**Terminal 2 — Frontend**
```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Project structure

```
ContextAgent/
│
├── docs/                        ← place your documents here
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py            ← PDF, TXT, MD loading
│   │   ├── chunker.py           ← smart chunking per file type
│   │   └── embedder.py          ← OpenAI embeddings
│   │
│   ├── storage/
│   │   └── vector_store.py      ← ChromaDB interface
│   │
│   ├── retrieval/
│   │   └── retriever.py         ← hybrid BM25 + vector + RRF
│   │
│   ├── generation/
│   │   ├── prompt.py            ← prompt templates
│   │   └── generator.py         ← LLM call + streaming
│   │
│   ├── memory/
│   │   └── buffer.py            ← conversation window buffer
│   │
│   └── agent/
│       └── orchestrator.py      ← coordinates the full pipeline
│
├── backend/
│   └── main.py                  ← FastAPI routes
│
├── frontend/
│   └── src/
│       ├── pages/               ← Chat, Dashboard, Documents, Evaluation
│       └── components/          ← Sidebar, shared components
│
├── scripts/
│   ├── ingest.py                ← run to ingest documents
│   ├── chat.py                  ← CLI interface (alternative to React)
│   └── evaluate_ragas.py        ← deterministic evaluation
│
├── results/                     ← evaluation JSON outputs
├── n8n/workflows/               ← n8n workflow exports
├── .env.example
└── requirements.txt
```

---

## Evaluation

Run the deterministic evaluation suite:

```bash
python scripts/evaluate_ragas.py --output results/eval.json
```

Current baseline (v2):

| Metric | Score |
|---|---|
| Keyword coverage | 0.8533 |
| Source accuracy | 1.0000 |
| Retrieval score | 0.5069 |
| Answer completeness | 1.0000 |
| Chunk coverage | 1.0000 |
| **Composite** | **0.8720** |

---

## Re-ingesting documents

To add new documents or change chunking parameters:

```bash
python scripts/ingest.py --reset
```

The `--reset` flag wipes ChromaDB and rebuilds from scratch.

---

## Roadmap

- v1 — RAG pipeline with CLI ✅
- v2 — Hybrid search, React interface, evaluation ✅
- v3 — Agent behavior, tool use, multi-document reasoning
- n8n — Visual workflow version