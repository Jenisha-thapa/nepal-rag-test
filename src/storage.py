"""
Conversation store — SQLite (Python standard library), file `data/conversations.db`.

    conversations(id, user_id, title, created_at, updated_at)
    messages(id, conversation_id, role, content, meta, created_at)   meta = JSON (sources, passages, ...)

There is no login: the web page creates a random user id and keeps it in the browser.
Every read/change checks that the conversation belongs to that user id.
The database is created on first use and is not committed (see .gitignore).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    meta             TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    # ── conversations ──────────────────────────────────────────────────────────
    def create(self, user_id: str, title: str) -> dict[str, Any]:
        conv = {"id": "c_" + uuid.uuid4().hex[:16], "user_id": user_id, "title": title.strip()[:80] or "New conversation",
                "created_at": _now()}
        conv["updated_at"] = conv["created_at"]
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO conversations VALUES (:id, :user_id, :title, :created_at, :updated_at)", conv)
        return {**conv, "message_count": 0}

    def list(self, user_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as db:
            rows = db.execute(
                """SELECT c.*, COUNT(m.id) AS message_count,
                          (SELECT json_extract(a.meta, '$.domain_id') FROM messages a
                            WHERE a.conversation_id = c.id AND a.role = 'assistant'
                            ORDER BY a.id DESC LIMIT 1) AS domain_id
                   FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
                   WHERE c.user_id = ? GROUP BY c.id ORDER BY c.updated_at DESC""", (user_id,)).fetchall()
        return [{**dict(r), "domain_id": r["domain_id"] or ""} for r in rows]

    def set_feedback(self, user_id: str, message_id: int, helpful: bool | None) -> bool:
        """Store 'was this answer helpful?' in the answer's meta (None clears it)."""
        with closing(self._connect()) as db, db:
            row = db.execute("""SELECT m.meta FROM messages m JOIN conversations c ON c.id = m.conversation_id
                                WHERE m.id = ? AND m.role = 'assistant' AND c.user_id = ?""",
                             (message_id, user_id)).fetchone()
            if row is None:
                return False
            meta = json.loads(row["meta"] or "{}")
            meta["helpful"] = helpful
            db.execute("UPDATE messages SET meta = ? WHERE id = ?", (json.dumps(meta, ensure_ascii=False), message_id))
        return True

    def get(self, user_id: str, conversation_id: str, with_messages: bool = True) -> dict[str, Any] | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                             (conversation_id, user_id)).fetchone()
            if row is None:
                return None
            conv = dict(row)
            if with_messages:
                conv["messages"] = [
                    {**dict(m), "meta": json.loads(m["meta"] or "{}")}
                    for m in db.execute("SELECT id, role, content, meta, created_at FROM messages "
                                        "WHERE conversation_id = ? ORDER BY id", (conversation_id,))
                ]
        return conv

    def rename(self, user_id: str, conversation_id: str, title: str) -> bool:
        with closing(self._connect()) as db, db:
            cur = db.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                             (title.strip()[:80], _now(), conversation_id, user_id))
        return cur.rowcount > 0

    def delete(self, user_id: str, conversation_id: str) -> bool:
        with closing(self._connect()) as db, db:
            cur = db.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id))
        return cur.rowcount > 0

    def delete_all(self, user_id: str) -> int:
        with closing(self._connect()) as db, db:
            return db.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,)).rowcount

    # ── messages ───────────────────────────────────────────────────────────────
    def add_message(self, conversation_id: str, role: str, content: str, meta: dict[str, Any] | None = None) -> int:
        now = _now()
        with closing(self._connect()) as db, db:
            cur = db.execute("INSERT INTO messages (conversation_id, role, content, meta, created_at) "
                             "VALUES (?, ?, ?, ?, ?)",
                             (conversation_id, role, content, json.dumps(meta or {}, ensure_ascii=False), now))
            db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
        return int(cur.lastrowid)

    def history(self, conversation_id: str, limit: int = 6) -> list[dict[str, str]]:
        """The last `limit` messages as [{"role", "content"}], oldest first (for follow-up questions)."""
        with closing(self._connect()) as db:
            rows = db.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                              (conversation_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]
