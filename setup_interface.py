"""
setup_interface.py
Crée toute l'arborescence backend FastAPI + frontend React
et écrit tous les fichiers de l'interface ContextAgent.

Usage:
    python setup_interface.py
"""

import os

def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  created: {path}")

# ═══════════════════════════════════════════════════════════════
# BACKEND
# ═══════════════════════════════════════════════════════════════

BACKEND_MAIN = '''"""
ContextAgent — FastAPI Backend
Exposes the RAG pipeline as a REST API consumed by the React frontend.
"""

import sys, time, uuid
from typing import Optional
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, ".")

from src.agent.orchestrator   import Orchestrator
from src.ingestion.loader     import load_all_documents
from src.ingestion.chunker    import chunk_documents
from src.ingestion.embedder   import embed_chunks
from src.storage.vector_store import (
    save_chunks, reset_collection, get_collection_count
)

app = FastAPI(title="ContextAgent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ─────────────────────────────────
sessions: dict[str, Orchestrator] = {}
session_stats: dict[str, dict]    = defaultdict(lambda: {
    "total_tokens": 0,
    "total_cost":   0.0,
    "turns":        0,
    "latencies":    [],
    "distances":    [],
    "token_history":[],
})

# ── Schemas ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question:   str
    top_k:      int   = 5
    temperature:float = 0.1
    memory_window:int = 6

class ResetRequest(BaseModel):
    session_id: str

class EvalRequest(BaseModel):
    questions: list[str]

# ── Helpers ─────────────────────────────────────────────────
def get_or_create_session(session_id: str, top_k: int = 5) -> Orchestrator:
    if session_id not in sessions:
        sessions[session_id] = Orchestrator(top_k=top_k, verbose=False)
    return sessions[session_id]

def compute_cost(tokens: int) -> float:
    return round((tokens / 1_000_000) * 0.15, 6)

# ── Routes ──────────────────────────────────────────────────

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

    return {
        "answer":   result["answer"],
        "sources":  result["sources"],
        "chunks":   result.get("chunks", []),
        "tokens":   result["tokens"],
        "model":    result["model"],
        "latency":  lat,
        "cost":     cost,
    }

@app.get("/api/documents")
def list_documents():
    import os
    from pathlib import Path
    docs_path = Path("docs/")
    supported = {".pdf", ".txt", ".md"}
    files = []
    if docs_path.exists():
        for f in sorted(docs_path.iterdir()):
            if f.suffix.lower() in supported:
                files.append({
                    "name":      f.name,
                    "type":      f.suffix.lower().replace(".", ""),
                    "size_kb":   round(f.stat().st_size / 1024, 1),
                    "path":      str(f),
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
            "success":    True,
            "filename":   file.filename,
            "chunks":     len(chunks),
            "total":      get_collection_count(),
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
    return {
        "success":   True,
        "documents": len(docs),
        "chunks":    len(chunks),
    }

@app.post("/api/reset")
def reset_chat(req: ResetRequest):
    if req.session_id in sessions:
        sessions[req.session_id].reset()
    session_stats[req.session_id] = {
        "total_tokens": 0,
        "total_cost":   0.0,
        "turns":        0,
        "latencies":    [],
        "distances":    [],
        "token_history":[],
    }
    return {"success": True}

@app.get("/api/stats/{session_id}")
def get_stats(session_id: str):
    s = session_stats[session_id]
    lats = s["latencies"]
    dsts = s["distances"]
    return {
        "total_tokens":   s["total_tokens"],
        "total_cost":     round(s["total_cost"], 6),
        "turns":          s["turns"],
        "avg_latency":    round(sum(lats)/len(lats), 2) if lats else 0,
        "avg_distance":   round(sum(dsts)/len(dsts), 4) if dsts else 0,
        "token_history":  s["token_history"],
        "distance_hist":  dsts,
        "total_chunks":   get_collection_count(),
    }

@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    """
    Run a quick evaluation on a list of questions.
    Returns per-question metrics: chunks found, avg distance, answer length.
    """
    from src.retrieval.retriever import retrieve
    results = []
    for q in req.questions:
        t0     = time.time()
        chunks = retrieve(q, top_k=5)
        lat    = round(time.time() - t0, 3)
        results.append({
            "question":     q,
            "chunks_found": len(chunks),
            "avg_distance": round(sum(c["distance"] for c in chunks)/len(chunks), 4) if chunks else 1.0,
            "sources":      list(set(c["source"] for c in chunks)),
            "latency":      lat,
        })
    return {"results": results}
'''

PACKAGE_JSON = '''{
  "name": "contextagent-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.22.0",
    "recharts": "^2.12.0",
    "axios": "^1.6.0",
    "react-dropzone": "^14.2.3",
    "react-markdown": "^9.0.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.0",
    "vite": "^5.1.0"
  }
}
'''

VITE_CONFIG = '''import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
'''

INDEX_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ContextAgent</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>
'''

MAIN_JSX = '''import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/global.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
'''

GLOBAL_CSS = ''':root {
  --bg:       #0c0c0f;
  --bg2:      #13131a;
  --bg3:      #1a1a24;
  --bg4:      #1e1e2e;
  --border:   #252535;
  --border2:  #30304a;
  --text:     #e2e2f0;
  --text2:    #8888a8;
  --text3:    #505070;
  --purple:   #7c6fe0;
  --purple2:  #a09be8;
  --purple3:  #c5c0f8;
  --purplebg: #1a1535;
  --purplebg2:#12102a;
  --teal:     #2dd4a0;
  --tealbg:   #0a2018;
  --amber:    #f0a830;
  --amberbg:  #1e1505;
  --red:      #f06060;
  --redbg:    #1e0808;
  --blue:     #60a0f0;
  --bluebg:   #080e1e;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  overflow: hidden;
}

#root { height: 100vh; display: flex; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--purple); }

button { cursor: pointer; font-family: inherit; }
input, textarea { font-family: inherit; }
'''

APP_JSX = '''import React, { useState } from \'react\'
import { BrowserRouter, Routes, Route, NavLink, useNavigate } from \'react-router-dom\'
import Chat from \'./pages/Chat\'
import Dashboard from \'./pages/Dashboard\'
import Documents from \'./pages/Documents\'
import Evaluation from \'./pages/Evaluation\'
import Sidebar from \'./components/Sidebar\'
import \'./styles/app.css\'

const NAV = [
  { path: \'/\',          icon: \'chat\',  label: \'Chat\' },
  { path: \'/dashboard\', icon: \'dash\',  label: \'Dashboard\' },
  { path: \'/documents\', icon: \'docs\',  label: \'Documents\' },
  { path: \'/eval\',      icon: \'eval\',  label: \'Evaluation\' },
]

function NavIcon({ type }) {
  if (type === \'chat\') return <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M3 5h12M3 9h12M3 13h8" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>
  if (type === \'dash\') return <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="3" y="3" width="5" height="5" rx="1.5" stroke="currentColor" strokeWidth="1.3"/><rect x="10" y="3" width="5" height="5" rx="1.5" stroke="currentColor" strokeWidth="1.3"/><rect x="3" y="10" width="5" height="5" rx="1.5" stroke="currentColor" strokeWidth="1.3"/><rect x="10" y="10" width="5" height="5" rx="1.5" stroke="currentColor" strokeWidth="1.3"/></svg>
  if (type === \'docs\') return <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M4 4h10a1 1 0 011 1v8a1 1 0 01-1 1H4a1 1 0 01-1-1V5a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.3"/><path d="M7 8l2 2 4-4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>
  if (type === \'eval\') return <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M3 14l3-3 3 3 3-5 3 5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/></svg>
  return null
}

function NavBar() {
  return (
    <nav className="navbar">
      <div className="nav-logo">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <circle cx="9" cy="9" r="6" stroke="#a09be8" strokeWidth="1.2"/>
          <path d="M5 9h8M9 5v8" stroke="#7c6fe0" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
      </div>
      {NAV.map(n => (
        <NavLink key={n.path} to={n.path} end={n.path === \'/\'} className={({isActive}) => \'nav-item\' + (isActive ? \' active\' : \'\')}>
          <NavIcon type={n.icon}/>
          <span className="nav-tooltip">{n.label}</span>
        </NavLink>
      ))}
    </nav>
  )
}

export default function App() {
  const [sessionId] = useState(() => \'sess_\' + Math.random().toString(36).slice(2))
  const [settings, setSettings] = useState({ topK: 5, temperature: 0.1, memoryWindow: 6 })
  const [debugMode, setDebugMode] = useState(true)

  const ctx = { sessionId, settings, setSettings, debugMode, setDebugMode }

  return (
    <BrowserRouter>
      <div style={{ display: \'flex\', height: \'100vh\', width: \'100%\' }}>
        <NavBar/>
        <Sidebar ctx={ctx}/>
        <main style={{ flex: 1, overflow: \'hidden\', display: \'flex\', flexDirection: \'column\' }}>
          <Routes>
            <Route path="/"          element={<Chat ctx={ctx}/>}/>
            <Route path="/dashboard" element={<Dashboard ctx={ctx}/>}/>
            <Route path="/documents" element={<Documents/>}/>
            <Route path="/eval"      element={<Evaluation ctx={ctx}/>}/>
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
'''

APP_CSS = '''.navbar {
  width: 56px; background: var(--bg2); border-right: 1px solid var(--border);
  display: flex; flex-direction: column; align-items: center;
  padding: 12px 0; gap: 4px; flex-shrink: 0;
}
.nav-logo {
  width: 34px; height: 34px; background: var(--purplebg);
  border: 1px solid var(--purple); border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 12px; cursor: pointer;
}
.nav-item {
  width: 38px; height: 38px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  color: var(--text3); text-decoration: none;
  transition: all 0.15s; position: relative;
}
.nav-item:hover { background: var(--bg3); color: var(--text2); }
.nav-item.active { background: var(--purplebg); color: var(--purple2); }
.nav-tooltip {
  position: absolute; left: 48px; background: var(--bg4);
  border: 1px solid var(--border2); border-radius: 6px;
  padding: 4px 10px; font-size: 11px; color: var(--text2);
  white-space: nowrap; pointer-events: none; opacity: 0;
  z-index: 100; transition: opacity 0.1s;
}
.nav-item:hover .nav-tooltip { opacity: 1; }
.topbar {
  height: 48px; background: var(--bg2); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 20px; flex-shrink: 0;
}
.topbar-title { font-size: 14px; font-weight: 600; color: var(--text); }
.topbar-right { display: flex; gap: 8px; align-items: center; }
.tbtn {
  font-size: 11px; padding: 5px 12px; border-radius: 8px;
  border: 1px solid var(--border2); background: transparent;
  color: var(--text2); transition: all 0.15s;
}
.tbtn:hover { background: var(--bg3); color: var(--text); }
.tbtn.on { background: var(--purplebg); color: var(--purple2); border-color: var(--purple); }
.sec-label {
  font-size: 9px; font-weight: 700; color: var(--text3);
  text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;
}
.stat-card {
  background: var(--bg3); border-radius: 10px; padding: 12px 14px;
  border: 1px solid var(--border); text-align: left;
}
.stat-value { font-size: 22px; font-weight: 600; color: var(--text); line-height: 1; }
.stat-value.green  { color: var(--teal); }
.stat-value.purple { color: var(--purple2); }
.stat-value.amber  { color: var(--amber); }
.stat-label { font-size: 10px; color: var(--text3); margin-top: 4px; }
.badge {
  font-size: 9px; padding: 2px 6px; border-radius: 4px;
  font-weight: 700; letter-spacing: 0.02em;
}
.badge-pdf { background: #2a1008; color: #c06040; }
.badge-txt { background: #081020; color: #4080c0; }
.badge-md  { background: #081508; color: #408040; }
.chip {
  font-size: 10px; padding: 2px 8px; border-radius: 20px;
  background: var(--tealbg); color: var(--teal);
  border: 1px solid #1a4030;
}
'''

SIDEBAR_JSX = '''import React, { useState, useEffect } from \'react\'
import axios from \'axios\'
import \'../styles/sidebar.css\'

export default function Sidebar({ ctx }) {
  const [docs, setDocs]       = useState([])
  const [chunks, setChunks]   = useState(0)
  const [stats, setStats]     = useState(null)

  useEffect(() => {
    axios.get(\'/api/documents\').then(r => {
      setDocs(r.data.documents)
      setChunks(r.data.total_chunks)
    })
  }, [])

  useEffect(() => {
    if (!ctx.sessionId) return
    const interval = setInterval(() => {
      axios.get(\`/api/stats/${ctx.sessionId}\`).then(r => setStats(r.data))
    }, 3000)
    return () => clearInterval(interval)
  }, [ctx.sessionId])

  const badge = ext => {
    if (ext === \'pdf\') return <span className="badge badge-pdf">PDF</span>
    if (ext === \'md\')  return <span className="badge badge-md">MD</span>
    return <span className="badge badge-txt">TXT</span>
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand">ContextAgent</div>
        <div className="sidebar-sub">Document assistant · v1</div>
        <div className={\'status-row \' + (chunks > 0 ? \'ready\' : \'empty\')}>
          <span className="status-dot"/>
          <span>{chunks > 0 ? \`${chunks} chunks ready\` : \'No chunks indexed\'}</span>
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sec-label">Documents</div>
        <div className="doc-list">
          {docs.map(d => (
            <div key={d.name} className="doc-item">
              <div className="doc-row">
                <span className="doc-name" title={d.name}>{d.name}</span>
                {badge(d.type)}
              </div>
              <div className="doc-meta">{d.size_kb} KB</div>
            </div>
          ))}
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sec-label">Session</div>
        <div className="stats-grid">
          <div className="stat-card"><div className="stat-value purple">{stats?.turns ?? 0}</div><div className="stat-label">Turns</div></div>
          <div className="stat-card"><div className="stat-value">{stats ? (stats.total_tokens >= 1000 ? (stats.total_tokens/1000).toFixed(1)+\'k\' : stats.total_tokens) : 0}</div><div className="stat-label">Tokens</div></div>
          <div className="stat-card"><div className="stat-value">{chunks}</div><div className="stat-label">Chunks</div></div>
          <div className="stat-card"><div className="stat-value green">${stats?.total_cost?.toFixed(4) ?? \'0.00\'}</div><div className="stat-label">Cost</div></div>
        </div>
      </div>

      <div className="sidebar-section">
        <div className="sec-label">Settings</div>
        <div className="slider-group">
          {[
            { label: \'Top K\', key: \'topK\', min: 1, max: 10 },
            { label: \'Memory\', key: \'memoryWindow\', min: 1, max: 12 },
          ].map(s => (
            <div key={s.key} className="slider-row">
              <div className="slider-header">
                <span className="slider-label">{s.label}</span>
                <span className="slider-val">{ctx.settings[s.key]}</span>
              </div>
              <input type="range" min={s.min} max={s.max} step="1"
                value={ctx.settings[s.key]}
                onChange={e => ctx.setSettings(p => ({...p, [s.key]: +e.target.value}))}/>
            </div>
          ))}
          <div className="slider-row">
            <div className="slider-header">
              <span className="slider-label">Temperature</span>
              <span className="slider-val">{ctx.settings.temperature.toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.05"
              value={ctx.settings.temperature}
              onChange={e => ctx.setSettings(p => ({...p, temperature: +e.target.value}))}/>
          </div>
        </div>
      </div>
    </aside>
  )
}
'''

SIDEBAR_CSS = '''.sidebar {
  width: 220px; background: var(--bg2); border-right: 1px solid var(--border);
  display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0;
}
.sidebar-header { padding: 16px; border-bottom: 1px solid var(--border); }
.sidebar-brand  { font-size: 14px; font-weight: 600; color: var(--text); }
.sidebar-sub    { font-size: 11px; color: var(--text3); margin-top: 2px; }
.status-row {
  display: flex; align-items: center; gap: 6px; margin-top: 10px;
  padding: 6px 10px; border-radius: 8px; font-size: 11px; border: 1px solid;
}
.status-row.ready  { background: var(--tealbg); color: var(--teal); border-color: #1a4030; }
.status-row.empty  { background: var(--redbg);  color: var(--red);  border-color: #4a1010; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
.sidebar-section { padding: 12px 16px; border-bottom: 1px solid var(--border); }
.doc-list { display: flex; flex-direction: column; gap: 4px; }
.doc-item {
  padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg3); cursor: default;
}
.doc-item:hover { border-color: var(--border2); }
.doc-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }
.doc-name { font-size: 11px; font-weight: 500; color: var(--text); max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-meta { font-size: 10px; color: var(--text3); }
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.slider-group { display: flex; flex-direction: column; gap: 12px; }
.slider-row { display: flex; flex-direction: column; gap: 5px; }
.slider-header { display: flex; justify-content: space-between; align-items: center; }
.slider-label { font-size: 11px; color: var(--text2); }
.slider-val { font-size: 11px; font-weight: 600; color: var(--purple2); }
input[type=range] { width: 100%; accent-color: var(--purple); height: 4px; }
'''

CHAT_JSX = '''import React, { useState, useRef, useEffect } from \'react\'
import axios from \'axios\'
import ReactMarkdown from \'react-markdown\'
import \'../styles/chat.css\'

function ChunkPanel({ chunks }) {
  if (!chunks.length) return (
    <div className="chunk-empty">No query yet — chunks will appear here</div>
  )
  return (
    <div className="chunk-list">
      {chunks.map((c, i) => {
        const scoreClass = c.distance < 0.45 ? \'hi\' : c.distance < 0.65 ? \'md\' : \'lo\'
        const pct = Math.round(c.distance * 100)
        return (
          <div key={i} className="chunk-card">
            <div className="chunk-top">
              <span className={\'score score-\' + scoreClass}>{c.distance.toFixed(3)}</span>
              <span className="chunk-src">{c.source} · #{c.chunk_index}</span>
            </div>
            <div className="chunk-txt">{c.text.slice(0, 240)}...</div>
            <div className="chunk-bar-bg">
              <div className={\'chunk-bar-fill \' + scoreClass} style={{ width: pct + \'%\' }}/>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default function Chat({ ctx }) {
  const [messages, setMessages]   = useState([])
  const [input, setInput]         = useState(\'\')
  const [loading, setLoading]     = useState(false)
  const [lastChunks, setChunks]   = useState([])
  const bottomRef = useRef(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: \'smooth\' }) }, [messages])

  const send = async () => {
    const q = input.trim()
    if (!q || loading) return
    setInput(\'\')
    setMessages(m => [...m, { role: \'user\', content: q }])
    setLoading(true)
    try {
      const { data } = await axios.post(\'/api/chat\', {
        session_id:    ctx.sessionId,
        question:      q,
        top_k:         ctx.settings.topK,
        temperature:   ctx.settings.temperature,
        memory_window: ctx.settings.memoryWindow,
      })
      setChunks(data.chunks || [])
      setMessages(m => [...m, {
        role: \'assistant\', content: data.answer,
        sources: data.sources, tokens: data.tokens.total,
        latency: data.latency, cost: data.cost,
      }])
    } catch(e) {
      setMessages(m => [...m, { role: \'error\', content: e.response?.data?.detail || \'API error\' }])
    }
    setLoading(false)
  }

  const reset = async () => {
    await axios.post(\'/api/reset\', { session_id: ctx.sessionId })
    setMessages([])
    setChunks([])
  }

  return (
    <div className="chat-page">
      <div className="topbar">
        <span className="topbar-title">Chat</span>
        <div className="topbar-right">
          <button className={\'tbtn\' + (ctx.debugMode ? \' on\' : \'\')} onClick={() => ctx.setDebugMode(d => !d)}>Debug</button>
          <button className="tbtn" onClick={reset}>Reset</button>
        </div>
      </div>

      <div className="chat-body">
        <div className="chat-main">
          <div className="chat-msgs">
            {messages.length === 0 && (
              <div className="chat-empty">
                <div className="chat-empty-icon">◈</div>
                <div className="chat-empty-title">Ask your documents</div>
                <div className="chat-empty-sub">Answers grounded in your content — always cited</div>
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={\'msg-row \' + m.role}>
                <div className={\'avatar av-\' + (m.role === \'user\' ? \'u\' : \'a\')}>{m.role === \'user\' ? \'S\' : \'CA\'}</div>
                <div className="msg-content">
                  <div className={\'bubble bubble-\' + m.role}>
                    {m.role === \'assistant\'
                      ? <ReactMarkdown>{m.content}</ReactMarkdown>
                      : m.content}
                  </div>
                  {m.role === \'assistant\' && (
                    <div className="msg-meta">
                      {m.sources?.map(s => <span key={s} className="chip">{s}</span>)}
                      <span className="meta-txt">{m.tokens} tokens · {m.latency}s · ${m.cost?.toFixed(5)}</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="msg-row assistant">
                <div className="avatar av-a">CA</div>
                <div className="bubble bubble-assistant">
                  <div className="typing"><span/><span/><span/></div>
                </div>
              </div>
            )}
            <div ref={bottomRef}/>
          </div>

          <div className="input-area">
            <div className="input-wrap">
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === \'Enter\' && !e.shiftKey && send()}
                placeholder="Ask a question about your documents..."
              />
            </div>
            <button className="send-btn" onClick={send} disabled={loading}>
              <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                <path d="M2 7.5h11M8 2.5l5 5-5 5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
        </div>

        {ctx.debugMode && (
          <div className="debug-panel">
            <div className="debug-header">
              <span className="debug-title">Retrieved chunks</span>
              <span className="debug-sub">{lastChunks.length} results</span>
            </div>
            <ChunkPanel chunks={lastChunks}/>
          </div>
        )}
      </div>
    </div>
  )
}
'''

CHAT_CSS = '''.chat-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.chat-body { flex: 1; display: grid; grid-template-columns: 1fr 260px; overflow: hidden; }
.chat-main { display: flex; flex-direction: column; overflow: hidden; }
.chat-msgs { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px; }
.chat-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 10px; }
.chat-empty-icon  { font-size: 36px; color: var(--purple); }
.chat-empty-title { font-size: 18px; font-weight: 600; color: var(--text); }
.chat-empty-sub   { font-size: 13px; color: var(--text3); }
.msg-row { display: flex; gap: 10px; align-items: flex-start; }
.msg-row.user { flex-direction: row-reverse; }
.avatar { width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 700; flex-shrink: 0; }
.av-a { background: var(--purplebg); color: var(--purple2); border: 1px solid var(--purple); }
.av-u { background: var(--bg3); color: var(--text2); border: 1px solid var(--border2); }
.msg-content { max-width: 80%; display: flex; flex-direction: column; gap: 5px; }
.bubble { padding: 11px 15px; font-size: 13px; line-height: 1.65; border-radius: 12px; }
.bubble-assistant { background: var(--bg3); border: 1px solid var(--border); border-radius: 4px 12px 12px 12px; color: var(--text); }
.bubble-user      { background: var(--purplebg); border: 1px solid var(--purple); border-radius: 12px 4px 12px 12px; color: var(--purple3); }
.bubble-error     { background: var(--redbg); border: 1px solid #4a1010; border-radius: 4px 12px 12px 12px; color: var(--red); }
.bubble p { margin: 0 0 8px; } .bubble p:last-child { margin: 0; }
.bubble code { background: var(--bg4); padding: 1px 5px; border-radius: 4px; font-size: 12px; color: var(--purple3); }
.bubble pre { background: var(--bg4); padding: 10px 12px; border-radius: 8px; overflow-x: auto; margin: 8px 0; }
.bubble pre code { background: none; padding: 0; color: var(--text2); }
.msg-meta { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.meta-txt { font-size: 10px; color: var(--text3); }
.input-area { padding: 12px 16px; border-top: 1px solid var(--border); background: var(--bg2); display: flex; gap: 8px; }
.input-wrap { flex: 1; background: var(--bg3); border: 1px solid var(--border2); border-radius: 10px; display: flex; align-items: center; padding: 0 14px; transition: border-color 0.15s; }
.input-wrap:focus-within { border-color: var(--purple); }
.input-wrap input { background: none; border: none; outline: none; color: var(--text); font-size: 13px; flex: 1; padding: 10px 0; }
.input-wrap input::placeholder { color: var(--text3); }
.send-btn { width: 38px; height: 38px; border-radius: 10px; background: var(--purple); border: none; display: flex; align-items: center; justify-content: center; transition: opacity 0.15s; }
.send-btn:hover { opacity: 0.85; }
.send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.typing { display: flex; gap: 4px; align-items: center; padding: 4px 0; }
.typing span { width: 6px; height: 6px; border-radius: 50%; background: var(--purple2); animation: blink 1.2s infinite; }
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%,80%,100%{opacity:0.3} 40%{opacity:1} }
.debug-panel { border-left: 1px solid var(--border); background: var(--bg2); display: flex; flex-direction: column; overflow: hidden; }
.debug-header { padding: 12px 14px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.debug-title { font-size: 12px; font-weight: 600; color: var(--text2); }
.debug-sub   { font-size: 10px; color: var(--text3); }
.chunk-list  { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 8px; }
.chunk-empty { padding: 24px 14px; font-size: 12px; color: var(--text3); text-align: center; }
.chunk-card  { background: var(--bg3); border: 1px solid var(--border); border-radius: 9px; padding: 10px 12px; }
.chunk-top   { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
.score { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 20px; }
.score-hi { background: var(--tealbg); color: var(--teal); border: 1px solid #1a4030; }
.score-md { background: var(--amberbg); color: var(--amber); border: 1px solid #4a3010; }
.score-lo { background: var(--redbg); color: var(--red); border: 1px solid #4a1010; }
.chunk-src { font-size: 10px; color: var(--text3); }
.chunk-txt { font-size: 11px; color: var(--text2); line-height: 1.5; margin-bottom: 6px; }
.chunk-bar-bg   { background: var(--border); border-radius: 3px; height: 3px; overflow: hidden; }
.chunk-bar-fill { height: 3px; border-radius: 3px; }
.chunk-bar-fill.hi { background: var(--teal); }
.chunk-bar-fill.md { background: var(--amber); }
.chunk-bar-fill.lo { background: var(--red); }
'''

DASHBOARD_JSX = '''import React, { useState, useEffect } from \'react\'
import axios from \'axios\'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis } from \'recharts\'
import \'../styles/dashboard.css\'

const TT = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: \'#1a1a24\', border: \'1px solid #252535\', borderRadius: 8, padding: \'8px 12px\', fontSize: 12, color: \'#e2e2f0\' }}>
      {payload.map((p,i) => <div key={i}>{p.name}: <b>{typeof p.value === \'number\' ? p.value.toFixed(3) : p.value}</b></div>)}
    </div>
  )
}

export default function Dashboard({ ctx }) {
  const [stats, setStats] = useState(null)
  const [docs,  setDocs]  = useState([])

  useEffect(() => {
    axios.get(\`/api/stats/${ctx.sessionId}\`).then(r => setStats(r.data))
    axios.get(\'/api/documents\').then(r => setDocs(r.data.documents))
    const iv = setInterval(() => {
      axios.get(\`/api/stats/${ctx.sessionId}\`).then(r => setStats(r.data))
    }, 5000)
    return () => clearInterval(iv)
  }, [])

  const tokenData = stats?.token_history?.map((t, i) => ({ turn: i+1, tokens: t })) || []
  const distData  = stats?.distance_hist?.map((d, i) => ({ i: i+1, distance: +d.toFixed(3) })) || []
  const docData   = docs.map(d => ({ name: d.name.slice(0,12), size: d.size_kb }))

  const cards = [
    { label: \'Total tokens\',  value: stats?.total_tokens?.toLocaleString() ?? \'0\',     cls: \'\' },
    { label: \'Total cost\',    value: \'$\'+(stats?.total_cost?.toFixed(5) ?? \'0.00000\'),  cls: \'green\' },
    { label: \'Turns\',         value: stats?.turns ?? 0,                                  cls: \'purple\' },
    { label: \'Avg latency\',   value: (stats?.avg_latency ?? 0)+\'s\',                     cls: \'amber\' },
    { label: \'Avg distance\',  value: stats?.avg_distance?.toFixed(3) ?? \'—\',            cls: \'\' },
    { label: \'Chunks indexed\',value: stats?.total_chunks ?? 0,                           cls: \'\' },
  ]

  return (
    <div className="dash-page">
      <div className="topbar"><span className="topbar-title">Dashboard</span></div>
      <div className="dash-body">
        <div className="cards-row">
          {cards.map(c => (
            <div key={c.label} className="stat-card">
              <div className={\'stat-value \'+c.cls}>{c.value}</div>
              <div className="stat-label">{c.label}</div>
            </div>
          ))}
        </div>

        <div className="charts-grid">
          <div className="chart-card">
            <div className="chart-title">Tokens per turn</div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={tokenData}>
                <XAxis dataKey="turn" stroke="#505070" tick={{ fontSize: 10 }}/>
                <YAxis stroke="#505070" tick={{ fontSize: 10 }}/>
                <Tooltip content={<TT/>}/>
                <Line type="monotone" dataKey="tokens" stroke="#7c6fe0" strokeWidth={2} dot={{ r: 3, fill: \'#7c6fe0\' }}/>
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="chart-title">Retrieval distances</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={distData}>
                <XAxis dataKey="i" stroke="#505070" tick={{ fontSize: 10 }}/>
                <YAxis domain={[0,1]} stroke="#505070" tick={{ fontSize: 10 }}/>
                <Tooltip content={<TT/>}/>
                <Bar dataKey="distance" fill="#2dd4a0" radius={[3,3,0,0]}/>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="chart-title">Documents by size (KB)</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={docData} layout="vertical">
                <XAxis type="number" stroke="#505070" tick={{ fontSize: 10 }}/>
                <YAxis dataKey="name" type="category" stroke="#505070" tick={{ fontSize: 10 }} width={80}/>
                <Tooltip content={<TT/>}/>
                <Bar dataKey="size" fill="#a09be8" radius={[0,3,3,0]}/>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="chart-title">System health</div>
            <ResponsiveContainer width="100%" height={180}>
              <RadarChart data={[
                { metric: \'Retrieval\',  value: stats ? Math.round((1-stats.avg_distance)*100) : 0 },
                { metric: \'Speed\',      value: stats ? Math.round(Math.max(0, 100-(stats.avg_latency*20))) : 0 },
                { metric: \'Coverage\',   value: stats ? Math.min(100, Math.round(stats.total_chunks/7)) : 0 },
                { metric: \'Usage\',      value: stats ? Math.min(100, stats.turns*10) : 0 },
                { metric: \'Economy\',    value: stats ? Math.round(Math.max(0,100-(stats.total_cost*10000))) : 100 },
              ]}>
                <PolarGrid stroke="#252535"/>
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: \'#8888a8\' }}/>
                <Radar dataKey="value" stroke="#7c6fe0" fill="#7c6fe0" fillOpacity={0.2}/>
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
'''

DASHBOARD_CSS = '''.dash-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.dash-body { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 20px; }
.cards-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }
.charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.chart-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.chart-title { font-size: 12px; font-weight: 600; color: var(--text2); margin-bottom: 12px; }
@media (max-width: 900px) { .cards-row { grid-template-columns: repeat(3,1fr); } .charts-grid { grid-template-columns: 1fr; } }
'''

DOCUMENTS_JSX = '''import React, { useState, useEffect, useCallback } from \'react\'
import axios from \'axios\'
import { useDropzone } from \'react-dropzone\'
import \'../styles/documents.css\'

export default function Documents() {
  const [docs,     setDocs]     = useState([])
  const [chunks,   setChunks]   = useState(0)
  const [uploading,setUploading]= useState(false)
  const [progress, setProgress] = useState(\'\')
  const [reingesting, setReing] = useState(false)

  const load = () => axios.get(\'/api/documents\').then(r => {
    setDocs(r.data.documents); setChunks(r.data.total_chunks)
  })

  useEffect(() => { load() }, [])

  const onDrop = useCallback(async files => {
    if (!files.length) return
    setUploading(true)
    for (const file of files) {
      setProgress(\`Ingesting ${file.name}...\`)
      const fd = new FormData()
      fd.append(\'file\', file)
      try {
        await axios.post(\'/api/ingest\', fd)
      } catch(e) {
        setProgress(\'Error: \' + (e.response?.data?.detail || e.message))
      }
    }
    setProgress(\'\')
    setUploading(false)
    load()
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { \'application/pdf\': [], \'text/plain\': [], \'text/markdown\': [] }
  })

  const reingest = async () => {
    setReing(true)
    await axios.post(\'/api/reingest\')
    setReing(false)
    load()
  }

  const badge = ext => {
    if (ext === \'pdf\') return <span className="badge badge-pdf">PDF</span>
    if (ext === \'md\')  return <span className="badge badge-md">MD</span>
    return <span className="badge badge-txt">TXT</span>
  }

  return (
    <div className="docs-page">
      <div className="topbar">
        <span className="topbar-title">Documents</span>
        <div className="topbar-right">
          <button className="tbtn" onClick={reingest} disabled={reingesting}>
            {reingesting ? \'Re-ingesting...\' : \'Re-ingest all\'}
          </button>
        </div>
      </div>

      <div className="docs-body">
        <div {...getRootProps()} className={\'dropzone\' + (isDragActive ? \' active\' : \'\')}>
          <input {...getInputProps()}/>
          <div className="drop-icon">↑</div>
          <div className="drop-title">{isDragActive ? \'Drop files here\' : \'Drag & drop documents\'}</div>
          <div className="drop-sub">PDF, TXT, MD — or click to browse</div>
          {uploading && <div className="drop-progress">{progress}</div>}
        </div>

        <div className="docs-stats">
          <div className="stat-card"><div className="stat-value">{docs.length}</div><div className="stat-label">Documents</div></div>
          <div className="stat-card"><div className="stat-value purple">{chunks}</div><div className="stat-label">Total chunks</div></div>
        </div>

        <div className="docs-table">
          <div className="table-header">
            <span>Name</span><span>Type</span><span>Size</span>
          </div>
          {docs.map(d => (
            <div key={d.name} className="table-row">
              <span className="td-name">{d.name}</span>
              <span>{badge(d.type)}</span>
              <span className="td-size">{d.size_kb} KB</span>
            </div>
          ))}
          {docs.length === 0 && <div className="table-empty">No documents found in docs/</div>}
        </div>
      </div>
    </div>
  )
}
'''

DOCUMENTS_CSS = '''.docs-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.docs-body { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
.dropzone {
  border: 1.5px dashed var(--border2); border-radius: 14px; padding: 36px;
  text-align: center; cursor: pointer; transition: all 0.2s; background: var(--bg2);
}
.dropzone:hover, .dropzone.active { border-color: var(--purple); background: var(--purplebg2); }
.drop-icon  { font-size: 28px; color: var(--purple2); margin-bottom: 8px; }
.drop-title { font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.drop-sub   { font-size: 12px; color: var(--text3); }
.drop-progress { margin-top: 10px; font-size: 12px; color: var(--teal); }
.docs-stats { display: flex; gap: 10px; }
.docs-table { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
.table-header { display: grid; grid-template-columns: 1fr 80px 80px; padding: 10px 16px; background: var(--bg3); font-size: 10px; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid var(--border); }
.table-row { display: grid; grid-template-columns: 1fr 80px 80px; padding: 11px 16px; border-bottom: 1px solid var(--border); align-items: center; font-size: 12px; }
.table-row:last-child { border-bottom: none; }
.table-row:hover { background: var(--bg3); }
.td-name { color: var(--text); font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.td-size { color: var(--text3); }
.table-empty { padding: 24px; text-align: center; font-size: 12px; color: var(--text3); }
'''

EVALUATION_JSX = '''import React, { useState } from \'react\'
import axios from \'axios\'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from \'recharts\'
import \'../styles/evaluation.css\'

const DEFAULT_QUESTIONS = [
  "What is a Python class?",
  "How does error handling work in Python?",
  "What is retrieval augmented generation?",
  "What is the attention mechanism in transformers?",
  "How do Python modules work?",
  "What are data structures in Python?",
  "How does inheritance work in Python?",
  "What is the difference between a list and a tuple?",
  "How does LangChain work?",
  "What are the key contributions of the RAG paper?",
]

export default function Evaluation({ ctx }) {
  const [questions, setQuestions] = useState(DEFAULT_QUESTIONS.join(\'\\n\'))
  const [results,   setResults]   = useState([])
  const [running,   setRunning]   = useState(false)

  const run = async () => {
    const qs = questions.split(\'\\n\').map(q => q.trim()).filter(Boolean)
    if (!qs.length) return
    setRunning(true)
    const { data } = await axios.post(\'/api/evaluate\', { questions: qs })
    setResults(data.results)
    setRunning(false)
  }

  const avgDist  = results.length ? (results.reduce((a,r) => a+r.avg_distance,0)/results.length).toFixed(3) : \'—\'
  const avgChunks= results.length ? (results.reduce((a,r) => a+r.chunks_found,0)/results.length).toFixed(1) : \'—\'
  const avgLat   = results.length ? (results.reduce((a,r) => a+r.latency,0)/results.length).toFixed(2) : \'—\'
  const coverage = results.length ? Math.round(results.filter(r=>r.chunks_found>0).length/results.length*100) : 0

  const radarData = [
    { metric: \'Coverage\',  value: coverage },
    { metric: \'Relevance\', value: results.length ? Math.round((1-parseFloat(avgDist))*100) : 0 },
    { metric: \'Speed\',     value: results.length ? Math.round(Math.max(0,100-(parseFloat(avgLat)*30))) : 0 },
    { metric: \'Depth\',     value: results.length ? Math.min(100,Math.round(parseFloat(avgChunks)/5*100)) : 0 },
  ]

  return (
    <div className="eval-page">
      <div className="topbar">
        <span className="topbar-title">Evaluation</span>
        <div className="topbar-right">
          <button className="tbtn on" onClick={run} disabled={running}>
            {running ? \'Running...\' : \'Run evaluation\'}
          </button>
        </div>
      </div>
      <div className="eval-body">
        <div className="eval-left">
          <div className="eval-card">
            <div className="eval-card-title">Test questions</div>
            <textarea
              className="eval-textarea"
              value={questions}
              onChange={e => setQuestions(e.target.value)}
              rows={12}
              placeholder="One question per line..."
            />
          </div>
          {results.length > 0 && (
            <div className="results-table">
              <div className="rtable-header">
                <span>Question</span><span>Chunks</span><span>Dist</span><span>Lat</span>
              </div>
              {results.map((r,i) => (
                <div key={i} className="rtable-row">
                  <span className="rq">{r.question.slice(0,40)}...</span>
                  <span className={\'rc \' + (r.chunks_found > 0 ? \'g\' : \'r\')}>{r.chunks_found}</span>
                  <span className={\'rd \' + (r.avg_distance < 0.5 ? \'g\' : r.avg_distance < 0.7 ? \'a\' : \'r\')}>{r.avg_distance.toFixed(3)}</span>
                  <span className="rl">{r.latency.toFixed(2)}s</span>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="eval-right">
          <div className="eval-card">
            <div className="eval-card-title">Summary metrics</div>
            <div className="metrics-grid">
              <div className="metric-box"><div className="metric-val">{coverage}%</div><div className="metric-lbl">Coverage</div></div>
              <div className="metric-box"><div className="metric-val">{avgDist}</div><div className="metric-lbl">Avg distance</div></div>
              <div className="metric-box"><div className="metric-val">{avgChunks}</div><div className="metric-lbl">Avg chunks</div></div>
              <div className="metric-box"><div className="metric-val">{avgLat}s</div><div className="metric-lbl">Avg latency</div></div>
            </div>
          </div>
          <div className="eval-card">
            <div className="eval-card-title">Quality radar</div>
            <ResponsiveContainer width="100%" height={220}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#252535"/>
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: \'#8888a8\' }}/>
                <Tooltip formatter={v => v+\'%\'}/>
                <Radar dataKey="value" stroke="#7c6fe0" fill="#7c6fe0" fillOpacity={0.25}/>
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
'''

EVALUATION_CSS = '''.eval-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.eval-body { flex: 1; overflow-y: auto; padding: 20px; display: grid; grid-template-columns: 1fr 320px; gap: 16px; }
.eval-left { display: flex; flex-direction: column; gap: 14px; }
.eval-right { display: flex; flex-direction: column; gap: 14px; }
.eval-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.eval-card-title { font-size: 12px; font-weight: 600; color: var(--text2); margin-bottom: 12px; }
.eval-textarea {
  width: 100%; background: var(--bg3); border: 1px solid var(--border2);
  border-radius: 8px; color: var(--text); font-size: 12px; line-height: 1.6;
  padding: 10px 12px; resize: vertical; outline: none; transition: border-color 0.15s;
}
.eval-textarea:focus { border-color: var(--purple); }
.metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.metric-box { background: var(--bg3); border-radius: 10px; padding: 14px; border: 1px solid var(--border); text-align: center; }
.metric-val { font-size: 24px; font-weight: 600; color: var(--purple2); }
.metric-lbl { font-size: 10px; color: var(--text3); margin-top: 4px; }
.results-table { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
.rtable-header { display: grid; grid-template-columns: 1fr 60px 70px 60px; padding: 8px 14px; background: var(--bg3); font-size: 9px; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid var(--border); }
.rtable-row { display: grid; grid-template-columns: 1fr 60px 70px 60px; padding: 9px 14px; border-bottom: 1px solid var(--border); font-size: 11px; align-items: center; }
.rtable-row:last-child { border-bottom: none; }
.rtable-row:hover { background: var(--bg3); }
.rq { color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rc, .rd { font-weight: 600; }
.rc.g, .rd.g { color: var(--teal); }
.rc.r, .rd.r { color: var(--red); }
.rd.a { color: var(--amber); }
.rl { color: var(--text3); }
'''

# ═══════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════

files = {
    "backend/main.py":                        BACKEND_MAIN,
    "frontend/package.json":                  PACKAGE_JSON,
    "frontend/vite.config.js":                VITE_CONFIG,
    "frontend/index.html":                    INDEX_HTML,
    "frontend/src/main.jsx":                  MAIN_JSX,
    "frontend/src/App.jsx":                   APP_JSX,
    "frontend/src/styles/global.css":         GLOBAL_CSS,
    "frontend/src/styles/app.css":            APP_CSS,
    "frontend/src/styles/sidebar.css":        SIDEBAR_CSS,
    "frontend/src/styles/chat.css":           CHAT_CSS,
    "frontend/src/styles/dashboard.css":      DASHBOARD_CSS,
    "frontend/src/styles/documents.css":      DOCUMENTS_CSS,
    "frontend/src/styles/evaluation.css":     EVALUATION_CSS,
    "frontend/src/components/Sidebar.jsx":    SIDEBAR_JSX,
    "frontend/src/pages/Chat.jsx":            CHAT_JSX,
    "frontend/src/pages/Dashboard.jsx":       DASHBOARD_JSX,
    "frontend/src/pages/Documents.jsx":       DOCUMENTS_JSX,
    "frontend/src/pages/Evaluation.jsx":      EVALUATION_JSX,
}

if __name__ == "__main__":
    print("Creating ContextAgent interface...\n")
    for path, content in files.items():
        write(path, content)
    print(f"\nDone. {len(files)} files created.")
    print("\nNext steps:")
    print("  1. pip install fastapi uvicorn python-multipart")
    print("  2. cd frontend && npm install")
    print("  3. Terminal 1: uvicorn backend.main:app --reload")
    print("  4. Terminal 2: cd frontend && npm run dev")
    print("  5. Open http://localhost:5173")