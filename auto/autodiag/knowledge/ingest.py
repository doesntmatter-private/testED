"""Turn PDFs, Markdown, and text files into chunks and index them.

Token counting is approximated as words * 1.3, which is close enough for
sizing chunks and needs no tokenizer dependency.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .. import config
from .store import KnowledgeStore

SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}
STEP_RE = re.compile(r"^\s*(\d+[.)]|step\s+\d+|[-*])\s+", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Section:
    page: int | None
    section: str | None
    text: str


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1


def parse_file(path: Path) -> tuple[str, list[Section]]:
    """Return (title, sections). Each section maps to a page or heading."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix in (".md", ".markdown"):
        return _parse_markdown(path)
    return path.stem, [Section(None, None, path.read_text(errors="ignore"))]


def _parse_pdf(path: Path) -> tuple[str, list[Section]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = None
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip() or None
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append(Section(page=i, section=None, text=text))
    return title or path.stem, sections


def _parse_markdown(path: Path) -> tuple[str, list[Section]]:
    lines = path.read_text(errors="ignore").splitlines()
    title = path.stem
    sections: list[Section] = []
    current_heading: str | None = None
    buf: list[str] = []

    def flush():
        if buf and "".join(buf).strip():
            sections.append(Section(None, current_heading, "\n".join(buf).strip()))
        buf.clear()

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and title == path.stem:
                title = heading
                continue
            flush()
            current_heading = heading
        else:
            buf.append(line)
    flush()
    return title, sections


def chunk_text(text: str, target: int = config.CHUNK_TOKENS, overlap: int = config.CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Split text into ~target-token chunks, preferring paragraph and step boundaries."""
    if _approx_tokens(text) <= target:
        return [text.strip()] if text.strip() else []

    # Split into atomic units: paragraphs, but keep numbered steps together
    # with their continuation lines.
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        lines = para.split("\n")
        cur: list[str] = []
        for line in lines:
            if STEP_RE.match(line) and cur:
                units.append("\n".join(cur))
                cur = [line]
            else:
                cur.append(line)
        if cur:
            units.append("\n".join(cur))

    chunks: list[str] = []
    cur_units: list[str] = []
    cur_tokens = 0
    for unit in units:
        ut = _approx_tokens(unit)
        if cur_tokens + ut > target and cur_units:
            chunks.append("\n\n".join(cur_units))
            # overlap: carry trailing units up to `overlap` tokens
            carry: list[str] = []
            carried = 0
            for u in reversed(cur_units):
                t = _approx_tokens(u)
                if carried + t > overlap:
                    break
                carry.insert(0, u)
                carried += t
            cur_units = carry
            cur_tokens = carried
        cur_units.append(unit)
        cur_tokens += ut
    if cur_units:
        chunks.append("\n\n".join(cur_units))
    return chunks


def ingest_path(store: KnowledgeStore, path: str | Path, force: bool = False) -> list[tuple[str, int, str]]:
    """Ingest a file or directory. Returns [(filename, n_chunks, status)]."""
    p = Path(path)
    files = sorted(f for f in (p.rglob("*") if p.is_dir() else [p]) if f.suffix.lower() in SUPPORTED)
    results: list[tuple[str, int, str]] = []
    for f in files:
        content_hash = hashlib.sha256(f.read_bytes()).hexdigest()
        if store.has_document_hash(content_hash) and not force:
            results.append((f.name, 0, "skipped (unchanged)"))
            continue
        title, sections = parse_file(f)
        doc_id = hashlib.sha1(str(f.resolve()).encode()).hexdigest()[:12]
        if force:
            store.remove_document(doc_id)
        rows: list[tuple[int, int | None, str | None, str]] = []
        pos = 0
        for sec in sections:
            for piece in chunk_text(sec.text):
                rows.append((pos, sec.page, sec.section, piece))
                pos += 1
        if not rows:
            results.append((f.name, 0, "no text extracted"))
            continue
        n = store.add_document(doc_id, title, str(f), content_hash, rows)
        results.append((f.name, n, "ingested"))
    return results
