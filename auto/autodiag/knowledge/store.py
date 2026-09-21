"""SQLite + FTS5 storage for manual chunks.

Schema:
  documents(doc_id, title, path, content_hash, ingested_at)
  chunks(chunk_id, doc_id, position, page, section, text)
  chunks_fts(text)  -- FTS5, external content table over chunks
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from ..models import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    ingested_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    page INTEGER,
    section TEXT,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
"""


class KnowledgeStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    # ---- documents -------------------------------------------------------

    def has_document_hash(self, content_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return row is not None

    def add_document(
        self,
        doc_id: str,
        title: str,
        path: str,
        content_hash: str,
        chunks: list[tuple[int, int | None, str | None, str]],
    ) -> int:
        """Insert a document and its chunks. chunks = (position, page, section, text)."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO documents(doc_id, title, path, content_hash, ingested_at) VALUES (?,?,?,?,?)",
                (doc_id, title, path, content_hash, time.time()),
            )
            self.conn.executemany(
                "INSERT INTO chunks(chunk_id, doc_id, position, page, section, text) VALUES (?,?,?,?,?,?)",
                [
                    (f"{doc_id}:{pos}", doc_id, pos, page, section, text)
                    for pos, page, section, text in chunks
                ],
            )
        return len(chunks)

    def remove_document(self, doc_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def list_documents(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT d.doc_id, d.title, d.path, d.ingested_at, COUNT(c.chunk_id) AS n_chunks
               FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id
               GROUP BY d.doc_id ORDER BY d.ingested_at"""
        ).fetchall()

    def count_chunks(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ---- search ------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[Chunk]:
        """BM25 search. `query` must already be a valid FTS5 expression."""
        if not query.strip():
            return []
        try:
            rows = self.conn.execute(
                """SELECT c.chunk_id, c.doc_id, d.title, c.page, c.section, c.text,
                          bm25(chunks_fts) AS score
                   FROM chunks_fts
                   JOIN chunks c ON c.rowid = chunks_fts.rowid
                   JOIN documents d ON d.doc_id = c.doc_id
                   WHERE chunks_fts MATCH ?
                   ORDER BY score
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS expression; treat as no results rather than crash.
            return []
        return [
            Chunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                title=r["title"],
                page=r["page"],
                section=r["section"],
                text=r["text"],
                score=-float(r["score"]),  # bm25() returns negative-is-better; flip so higher is better
            )
            for r in rows
        ]

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"""SELECT c.chunk_id, c.doc_id, d.title, c.page, c.section, c.text
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.chunk_id IN ({marks})""",
            chunk_ids,
        ).fetchall()
        return [Chunk(**{k: r[k] for k in r.keys()}) for r in rows]

    def close(self) -> None:
        self.conn.close()
