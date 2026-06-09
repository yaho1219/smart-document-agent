"""SQLite 영속화.

처리 이력(원본 파일, OCR 원문, 분류 결과, 구조화 JSON)을 로컬 DB에
저장한다. 외부 전송 없이 노트북 안에 모든 데이터를 보관한다.

담당: Dev Engineer
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

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
    """테이블을 생성한다(존재 시 무시)."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def save_result(result: DocumentResult) -> int:
    """처리 결과 1건을 저장하고 row id를 반환한다."""
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
    """저장된 처리 이력을 최신순으로 반환한다."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def clear_all() -> None:
    """모든 이력을 삭제한다(데모 초기화용)."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM documents")
        conn.commit()
