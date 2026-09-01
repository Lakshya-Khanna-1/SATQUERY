"""
SatQuery AI — Local System SQLite Persistent Memory Engine
Database Path: data/satquery_memory.db

Provides persistent local storage for:
- Chat Sessions (id, title, is_pinned, created_at, updated_at)
- Messages (id, session_id, role, content, images, confidence, task_type, analysis_mode, evidence_image, trace, timestamp)
"""

import os
import sqlite3
import json
import time
import uuid
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("SatQueryDB")

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "satquery_memory.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite database tables for persistent local memory."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        is_pinned INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        images TEXT,
        confidence REAL,
        task_type TEXT,
        analysis_mode TEXT,
        evidence_image TEXT,
        trace TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()
    logger.info(f"SatQuery SQLite memory database initialized at: {DB_PATH}")


# Automatically initialize on module load
init_db()


def create_session(title: str = "New Analysis Session") -> Dict[str, Any]:
    session_id = f"session_{int(time.time()*1000)}"
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (id, title, is_pinned, created_at, updated_at) VALUES (?, ?, 0, ?, ?)",
        (session_id, title, now, now)
    )
    conn.commit()
    conn.close()
    return {"id": session_id, "title": title, "is_pinned": 0, "created_at": now, "updated_at": now, "messages": []}


def get_all_sessions() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions ORDER BY is_pinned DESC, updated_at DESC")
    rows = cursor.fetchall()
    
    sessions = []
    for r in rows:
        s_dict = dict(r)
        # Fetch last message content for subtitle preview
        cursor.execute(
            "SELECT content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT 1",
            (s_dict["id"],)
        )
        first_msg = cursor.fetchone()
        s_dict["subtitle"] = first_msg["content"] if first_msg else "Empty session"
        sessions.append(s_dict)
        
    conn.close()
    return sessions


def get_session_detail(session_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    s_row = cursor.fetchone()
    if not s_row:
        conn.close()
        return None

    session = dict(s_row)
    cursor.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY rowid ASC", (session_id,))
    m_rows = cursor.fetchall()

    messages = []
    for m in m_rows:
        m_dict = dict(m)
        if m_dict.get("images"):
            try:
                m_dict["images"] = json.loads(m_dict["images"])
            except Exception:
                pass
        if m_dict.get("trace"):
            try:
                m_dict["trace"] = json.loads(m_dict["trace"])
            except Exception:
                pass
        messages.append(m_dict)

    session["messages"] = messages
    conn.close()
    return session


def save_message(session_id: str, msg: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()

    # Verify session exists, if not create
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    s_row = cursor.fetchone()
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    if not s_row:
        first_prompt = msg.get("content", "Satellite Analysis")[:35]
        title = first_prompt if first_prompt else "Satellite Analysis"
        cursor.execute(
            "INSERT INTO sessions (id, title, is_pinned, created_at, updated_at) VALUES (?, ?, 0, ?, ?)",
            (session_id, title, now, now)
        )
    else:
        # If this is user prompt and session title is default, update session title
        if msg.get("role") == "user" and (s_row["title"] == "New Analysis Session" or s_row["title"] == "Satellite Analysis"):
            new_title = msg.get("content", "Satellite Analysis")[:35]
            cursor.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (new_title, now, session_id))
        else:
            cursor.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))

    msg_id = str(msg.get("id", f"msg_{int(time.time()*1000)}"))
    role = msg.get("role", "user")
    content = msg.get("content", "")
    images_json = json.dumps(msg.get("images", [])) if msg.get("images") else None
    confidence = msg.get("confidence")
    task_type = msg.get("task_type")
    analysis_mode = msg.get("analysis_mode")
    evidence_image = msg.get("evidence_image")
    trace_json = json.dumps(msg.get("trace", {})) if msg.get("trace") else None
    timestamp = msg.get("timestamp", now)

    cursor.execute("""
    INSERT INTO messages (id, session_id, role, content, images, confidence, task_type, analysis_mode, evidence_image, trace, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (msg_id, session_id, role, content, images_json, confidence, task_type, analysis_mode, evidence_image, trace_json, timestamp))

    conn.commit()
    conn.close()
    return msg


def toggle_pin_session(session_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT is_pinned FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    new_pinned = 0 if row["is_pinned"] == 1 else 1
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("UPDATE sessions SET is_pinned = ?, updated_at = ? WHERE id = ?", (new_pinned, now, session_id))
    conn.commit()
    conn.close()
    return {"id": session_id, "is_pinned": new_pinned}


def delete_session(session_id: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return True
