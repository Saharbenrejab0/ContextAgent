"""
ContextAgent — FastAPI Backend v2.1
SQLite memory persistence added.
All routes defined AFTER app = FastAPI().
"""

import sys, time, uuid, json, os
from pathlib import Path
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
    init_db,
    create_session,
    save_message,
    get_session_messages,
    get_all_sessions,
    delete_session,
)

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(title="ContextAgent API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ──────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    init_db()
    print("✅ SQLite memory database initialised.")

# ── In-memory session store ──────────────────────────────────────
sessions: dict[str, Orchestrator] = {}
session_stats: dict[str, dict] = defaultdict(lambda: {
    "total_tokens": 0,
    "total_cost":   0.0,
    "turns":        0,
    "latencies":    [],
    "distances":    [],
    "token_history": [],
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
def get_or_create_orchestrator(session_id: str, top_k: int = 5) -> Orchestrator:
    if session_id not in sessions:
        sessions[session_id] = Orchestrator(top_k=top_k, verbose=False)
    return sessions[session_id]

def compute_cost(tokens: int) -> float:
    return round((tokens / 1_000_000) * 0.15, 6)

# ════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════

# ── Health ───────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": get_collection_count()}


# ── Chat (non-streaming) ─────────────────────────────────────────
@app.post("/api/chat")
def chat(req: ChatRequest):
    if get_collection_count() == 0:
        raise HTTPException(400, "No documents ingested. Run ingestion first.")

    orch = get_or_create_orchestrator(req.session_id, req.top_k)
    t0   = time.time()

    result  = orch.run(req.question)
    latency = round(time.time() - t0, 2)
    tokens  = result.get("tokens_used", 0)
    cost    = compute_cost(tokens)

    # Update in-memory stats
    s = session_stats[req.session_id]
    s["total_tokens"]  += tokens
    s["total_cost"]    += cost
    s["turns"]         += 1
    s["latencies"].append(latency)
    s["token_history"].append({"turn": s["turns"], "tokens": tokens})
    s["distances"].extend([c.get("distance", 0) for c in result.get("chunks", [])])

    # Persist to SQLite
    create_session(req.session_id)
    save_message(req.session_id, "user",      req.question)
    save_message(req.session_id, "assistant", result.get("answer", ""), tokens=tokens)

    return {
        "answer":     result.get("answer", ""),
        "chunks":     result.get("chunks", []),
        "tokens":     tokens,
        "cost":       cost,
        "latency":    latency,
        "session_id": req.session_id,
    }


# ── Chat (streaming) ─────────────────────────────────────────────
@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    if get_collection_count() == 0:
        raise HTTPException(400, "No documents ingested. Run ingestion first.")

    orch = get_or_create_orchestrator(req.session_id, req.top_k)

    def event_generator():
        t0          = time.time()
        full_answer = []

        try:
            # Phase 1 — retrieval
            retrieval_result = orch.retrieve(req.question)
            chunks = retrieval_result.get("chunks", [])
            yield f"data: {json.dumps({'type': 'chunks', 'chunks': chunks})}\n\n"

            # Phase 2 — stream tokens
            for token in orch.stream_generate(req.question, chunks):
                full_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            answer  = "".join(full_answer)
            latency = round(time.time() - t0, 2)
            tokens  = orch.last_token_count()
            cost    = compute_cost(tokens)

            # Update in-memory stats
            s = session_stats[req.session_id]
            s["total_tokens"]  += tokens
            s["total_cost"]    += cost
            s["turns"]         += 1
            s["latencies"].append(latency)
            s["token_history"].append({"turn": s["turns"], "tokens": tokens})
            s["distances"].extend([c.get("distance", 0) for c in chunks])

            # Persist to SQLite
            create_session(req.session_id)
            save_message(req.session_id, "user",      req.question)
            save_message(req.session_id, "assistant", answer, tokens=tokens)

            yield f"data: {json.dumps({'type': 'done', 'tokens': tokens, 'cost': cost, 'latency': latency})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── SQLite Session Routes ────────────────────────────────────────

@app.get("/api/sessions")
def list_all_sessions():
    return {"sessions": get_all_sessions()}


@app.get("/api/sessions/{session_id}/history")
def get_history(session_id: str):
    history = get_session_messages(session_id)
    return {"session_id": session_id, "history": history}


@app.delete("/api/sessions/{session_id}")
def remove_session(session_id: str):
    delete_session(session_id)
    if session_id in sessions:
        del sessions[session_id]
    if session_id in session_stats:
        del session_stats[session_id]
    return {"success": True, "session_id": session_id}


# ── Documents ────────────────────────────────────────────────────
@app.get("/api/documents")
def list_documents():
    docs_dir = Path("docs")
    files    = []
    if docs_dir.exists():
        for f in docs_dir.iterdir():
            if f.suffix.lower() in {".pdf", ".txt", ".md"}:
                files.append({
                    "name":    f.name,
                    "type":    f.suffix.lstrip(".").upper(),
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "path":    str(f),
                })
    return {"documents": files, "total_chunks": get_collection_count()}


@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    import shutil
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
    """Clears in-memory context. Keeps SQLite history intact."""
    if req.session_id in sessions:
        sessions[req.session_id].reset()
    session_stats[req.session_id] = {
        "total_tokens": 0,
        "total_cost":   0.0,
        "turns":        0,
        "latencies":    [],
        "distances":    [],
        "token_history": [],
    }
    return {"success": True}


# ── Stats ────────────────────────────────────────────────────────
@app.get("/api/stats/{session_id}")
def get_stats(session_id: str):
    s    = session_stats[session_id]
    lats = s["latencies"]
    dsts = s["distances"]
    return {
        "total_tokens":  s["total_tokens"],
        "total_cost":    round(s["total_cost"], 6),
        "turns":         s["turns"],
        "avg_latency":   round(sum(lats) / len(lats), 2) if lats else 0,
        "avg_distance":  round(sum(dsts) / len(dsts), 4) if dsts else 0,
        "token_history": s["token_history"],
        "distances":     dsts,
    }


# ── Evaluation ───────────────────────────────────────────────────
@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    if get_collection_count() == 0:
        raise HTTPException(400, "No documents ingested.")

    orch    = Orchestrator(top_k=5, verbose=False)
    results = []

    for q in req.questions:
        t0     = time.time()
        result = orch.run(q)
        results.append({
            "question": q,
            "answer":   result.get("answer", ""),
            "chunks":   len(result.get("chunks", [])),
            "latency":  round(time.time() - t0, 2),
            "tokens":   result.get("tokens_used", 0),
        })

    return {"results": results, "total": len(results)}