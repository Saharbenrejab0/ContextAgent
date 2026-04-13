"""
ContextAgent — FastAPI Backend v2.1
SQLite memory persistence. All routes after app = FastAPI().
"""

import sys, time, uuid, json, os
from typing import Optional
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, ".")

from src.agent.orchestrator   import Orchestrator
from src.ingestion.loader     import load_all_documents
from src.ingestion.chunker    import chunk_documents
from src.ingestion.embedder   import embed_chunks
from src.storage.vector_store import (
    save_chunks, reset_collection, get_collection_count
)
from src.memory.database import (
    init_db, create_session, save_message,
    get_session_messages, get_all_sessions, delete_session
)

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(title="ContextAgent API", version="2.1.0")

init_db()
print("✅ SQLite memory database initialised.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ──────────────────────────────────────
sessions: dict[str, Orchestrator] = {}
session_stats: dict[str, dict]    = defaultdict(lambda: {
    "total_tokens": 0,
    "total_cost":   0.0,
    "turns":        0,
    "latencies":    [],
    "distances":    [],
    "token_history":[],
})

# ── Schemas ──────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id:    str
    question:      str
    top_k:         int   = 5
    temperature:   float = 0.1
    memory_window: int   = 6

class ResetRequest(BaseModel):
    session_id: str

class EvalRequest(BaseModel):
    questions: list[str]

# ── Helpers ──────────────────────────────────────────────────────
def get_or_create_session(session_id: str, top_k: int = 5) -> Orchestrator:
    if session_id not in sessions:
        sessions[session_id] = Orchestrator(top_k=top_k, verbose=False)
    return sessions[session_id]

def compute_cost(tokens: int) -> float:
    return round((tokens / 1_000_000) * 0.15, 6)

# ════════════════════════════════════════════════════════════════
# ROUTES — all defined after app = FastAPI()
# ════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": get_collection_count()}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if get_collection_count() == 0:
        raise HTTPException(400, "No documents ingested. Run ingestion first.")

    agent = get_or_create_session(req.session_id, req.top_k)
    agent.buffer.window = req.memory_window

    t0     = time.time()
    result = agent.ask(req.question)
    lat    = round(time.time() - t0, 2)

    tokens = result["tokens"]["total"]
    cost   = compute_cost(tokens)
    stats  = session_stats[req.session_id]

    stats["total_tokens"] += tokens
    stats["total_cost"]   += cost
    stats["turns"]        += 1
    stats["latencies"].append(lat)
    stats["token_history"].append(tokens)
    for c in result.get("chunks", []):
        stats["distances"].append(c["distance"])

    create_session(req.session_id)
    save_message(req.session_id, "user",      req.question)
    save_message(req.session_id, "assistant", result["answer"], tokens=tokens)

    return {
        "answer":   result["answer"],
        "sources":  result["sources"],
        "chunks":   result.get("chunks", []),
        "tokens":   result["tokens"],
        "model":    result["model"],
        "latency":  lat,
        "cost":     cost,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    if get_collection_count() == 0:
        raise HTTPException(400, "No documents ingested.")

    from src.retrieval.retriever  import retrieve, format_context
    from src.generation.prompt    import build_messages, build_standalone_question
    from src.generation.generator import generate_stream

    agent    = get_or_create_session(req.session_id, req.top_k)
    agent.buffer.window = req.memory_window
    history  = agent.buffer.get()
    search_q = build_standalone_question(req.question, history)
    chunks   = retrieve(
        query          = req.question,
        top_k          = req.top_k,
        original_query = req.question,
    )
    context  = format_context(chunks)
    messages = build_messages(
        question = req.question,
        context  = context,
        history  = history if history else None,
    )

    def event_stream():
        full = ""
        for token in generate_stream(messages):
            if token.startswith("[DONE]"):
                meta   = json.loads(token[6:])
                tokens = meta["tokens"]["total"]
                cost   = compute_cost(tokens)

                agent.buffer.add(req.question, full)

                create_session(req.session_id)
                save_message(req.session_id, "user", req.question)
                save_message(req.session_id, "assistant", full,
                             sources=meta.get("sources", []),
                             tokens=tokens)

                stats = session_stats[req.session_id]
                stats["total_tokens"] += tokens
                stats["total_cost"]   += cost
                stats["turns"]        += 1
                stats["token_history"].append(tokens)
                for c in chunks:
                    stats["distances"].append(c["distance"])

                meta["chunks"] = chunks
                meta["cost"]   = cost
                yield f"data: [DONE]{json.dumps(meta)}\n\n"
            else:
                full += token
                yield f"data: {json.dumps(token)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── SQLite Session Routes ────────────────────────────────────────

@app.get("/api/sessions")
def list_all_sessions():
    return {"sessions": get_all_sessions()}

@app.post("/api/sessions/{session_id}")
def create_new_session(session_id: str):
    return create_session(session_id)

@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str):
    msgs = get_session_messages(session_id, limit=100)
    return {"messages": msgs}

@app.delete("/api/sessions/{session_id}")
def remove_session(session_id: str):
    delete_session(session_id)
    if session_id in sessions:
        del sessions[session_id]
    if session_id in session_stats:
        del session_stats[session_id]
    return {"success": True}


# ── Documents ────────────────────────────────────────────────────

@app.get("/api/documents")
def list_documents():
    from pathlib import Path
    docs_path = Path("docs/")
    supported = {".pdf", ".txt", ".md"}
    files = []
    if docs_path.exists():
        for f in sorted(docs_path.iterdir()):
            if f.suffix.lower() in supported:
                files.append({
                    "name":    f.name,
                    "type":    f.suffix.lower().replace(".", ""),
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "path":    str(f),
                })
    return {"documents": files, "total_chunks": get_collection_count()}


@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    import shutil
    from pathlib import Path

    allowed = {".pdf", ".txt", ".md"}
    ext     = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    save_path = Path("docs") / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        docs     = load_all_documents(str(save_path.parent))
        target   = [d for d in docs if d["source"] == file.filename]
        if not target:
            raise HTTPException(500, "Failed to load document")
        chunks   = chunk_documents(target)
        embedded = embed_chunks(chunks)
        save_chunks(embedded)
        return {
            "success":  True,
            "filename": file.filename,
            "chunks":   len(chunks),
            "total":    get_collection_count(),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/reingest")
def reingest_all():
    reset_collection()
    docs     = load_all_documents("docs/")
    chunks   = chunk_documents(docs)
    embedded = embed_chunks(chunks)
    save_chunks(embedded)
    return {"success": True, "documents": len(docs), "chunks": len(chunks)}


# ── Reset ────────────────────────────────────────────────────────

@app.post("/api/reset")
def reset_chat(req: ResetRequest):
    if req.session_id in sessions:
        sessions[req.session_id].reset()
    session_stats[req.session_id] = {
        "total_tokens": 0, "total_cost": 0.0, "turns": 0,
        "latencies": [], "distances": [], "token_history": [],
    }
    return {"success": True}


# ── Stats ────────────────────────────────────────────────────────

@app.get("/api/stats/{session_id}")
def get_stats(session_id: str):
    try:
        total_chunks = get_collection_count()
    except Exception:
        total_chunks = 0
    s    = session_stats[session_id]
    lats = s["latencies"]
    dsts = s["distances"]
    return {
        "total_tokens":  s["total_tokens"],
        "total_cost":    round(s["total_cost"], 6),
        "turns":         s["turns"],
        "avg_latency":   round(sum(lats)/len(lats), 2) if lats else 0,
        "avg_distance":  round(sum(dsts)/len(dsts), 4) if dsts else 0,
        "token_history": s["token_history"],
        "distance_hist": dsts,
        "total_chunks":  total_chunks,
    }


# ── Evaluation ───────────────────────────────────────────────────

@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    from src.retrieval.retriever import retrieve
    results = []
    for q in req.questions:
        t0     = time.time()
        chunks = retrieve(q, top_k=5)
        lat    = round(time.time() - t0, 3)
        results.append({
            "question":     q,
            "chunks_found": len(chunks),
            "avg_distance": round(
                sum(c["distance"] for c in chunks) / len(chunks), 4
            ) if chunks else 1.0,
            "sources":  list(set(c["source"] for c in chunks)),
            "latency":  lat,
        })
    return {"results": results}