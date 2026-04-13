"""
SQLite memory persistence.
Saves every conversation turn to disk.
Sessions survive browser refresh and server restarts.
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path

DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/memory.db")


def get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                title      TEXT,
                created_at TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role       TEXT,
                content    TEXT,
                sources    TEXT,
                tokens     INTEGER,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)


def create_session(session_id: str, title: str = "New conversation") -> dict:
    now = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (session_id, title, now, now)
        )
    return {"id": session_id, "title": title, "created_at": now}


def save_message(
    session_id: str,
    role:       str,
    content:    str,
    sources:    list[str] = None,
    tokens:     int       = 0,
) -> None:
    now = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO messages
               (session_id, role, content, sources, tokens, created_at)
               VALUES (?,?,?,?,?,?)""",
            (session_id, role, content,
             ",".join(sources) if sources else "", tokens, now)
        )
        conn.execute(
            "UPDATE sessions SET updated_at=?, title=? WHERE id=?",
            (now, _make_title(content, role), session_id)
        )


def get_session_messages(session_id: str, limit: int = 12) -> list[dict]:
    """Return last N messages for a session as list of dicts."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT role, content, sources, tokens, created_at
               FROM messages
               WHERE session_id = ?
               ORDER BY id DESC LIMIT ?""",
            (session_id, limit)
        ).fetchall()
    rows.reverse()
    return [dict(r) for r in rows]


def get_all_sessions(limit: int = 20) -> list[dict]:
    """Return most recent sessions for session history sidebar."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, title, created_at, updated_at
               FROM sessions
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))


def _make_title(content: str, role: str) -> str:
    """Use first user message as session title."""
    if role != "user":
        return "Conversation"
    return content[:50] + ("..." if len(content) > 50 else "")