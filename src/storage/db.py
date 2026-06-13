from __future__ import annotations

import json
import sqlite3

from src.config import ensure_dir, load_config, resolve_path
from src.schemas import DocumentResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file   TEXT NOT NULL,
    doc_type      TEXT NOT NULL,
    confidence    REAL,
    ocr_text      TEXT,
    structured    TEXT,
    processed_at  TEXT NOT NULL
);
"""


def _db_path() -> str:
    rel = load_config().get("paths", {}).get("database", "data/documents.db")
    path = resolve_path(rel)
    ensure_dir(path.parent)
    return str(path)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def save_result(result: DocumentResult) -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO documents
                (source_file, doc_type, confidence, ocr_text, structured, processed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result.source_file,
                result.classification.doc_type.value,
                result.classification.confidence,
                result.ocr.full_text,
                json.dumps(result.structured_dict(), ensure_ascii=False),
                result.processed_at.isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def fetch_all(limit: int = 200) -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def clear_all() -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM documents")
        conn.commit()
