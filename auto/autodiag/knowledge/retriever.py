"""Deterministic query construction plus BM25 search and rank fusion.

No LLM is involved here, by design (see DESIGN.md 4.6, stage 2-3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import config
from ..models import Chunk, Flag, VehicleSnapshot
from .store import KnowledgeStore

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "is", "it", "its", "my", "car",
    "when", "with", "for", "i", "im", "i'm", "has", "have", "was", "were", "been", "be", "this",
    "that", "there", "then", "than", "also", "very", "just", "but", "so", "as", "after", "before",
    "while", "during", "from", "into", "up", "down", "out", "off", "over", "under", "again",
    "sometimes", "always", "never", "seems", "seem", "like", "feels", "feel", "get", "gets",
    "getting", "goes", "going", "still", "yet", "now", "since", "about", "some", "any", "all",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


@dataclass
class Query:
    kind: str  # "dtc" | "symptoms" | "flag" | "vehicle" | "combined"
    text: str
    weight: float = 1.0


def _fts_terms(text: str) -> list[str]:
    """Tokenize free text into safe FTS5 terms (quoted, stopwords removed)."""
    terms = []
    for tok in TOKEN_RE.findall(text):
        t = tok.lower().strip("'-")
        if len(t) < 2 or t in STOPWORDS:
            continue
        terms.append(t)
    # dedupe preserving order
    seen: set[str] = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _or_query(terms: list[str]) -> str:
    """Build an OR-query: any term matches, BM25 ranks by how many."""
    return " OR ".join(f'"{t}"' for t in terms)


def build_queries(snapshot: VehicleSnapshot, symptoms: str = "", flags: list[Flag] | None = None) -> list[Query]:
    queries: list[Query] = []
    vehicle_terms = _fts_terms(
        " ".join(filter(None, [snapshot.vehicle.make, snapshot.vehicle.model, snapshot.vehicle.engine]))
    )

    # One per DTC: exact code required, description words optional.
    for dtc in snapshot.dtcs:
        desc_terms = _fts_terms(dtc.description)[:8]
        q = f'"{dtc.code.lower()}"'
        if desc_terms:
            q = f'({q} OR {_or_query(desc_terms)})'
        queries.append(Query("dtc", q, weight=config.DTC_QUERY_WEIGHT))

    symptom_terms = _fts_terms(symptoms)[:15]
    if symptom_terms:
        queries.append(Query("symptoms", _or_query(symptom_terms)))

    for flag in flags or []:
        terms = _fts_terms(flag.search_terms)
        if terms:
            queries.append(Query("flag", _or_query(terms)))

    if vehicle_terms:
        queries.append(Query("vehicle", _or_query(vehicle_terms), weight=0.5))

    combined = [d.code.lower() for d in snapshot.dtcs] + symptom_terms[:8] + vehicle_terms
    if combined:
        queries.append(Query("combined", _or_query(combined)))

    return queries


class Retriever:
    def __init__(self, store: KnowledgeStore, top_k: int = config.TOP_K, per_query: int = config.PER_QUERY_LIMIT):
        self.store = store
        self.top_k = top_k
        self.per_query = per_query

    def search(self, queries: list[Query]) -> list[Chunk]:
        """Run every query, fuse with weighted RRF, collapse same-page hits, assign handles."""
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for q in queries:
            results = self.store.search(q.text, limit=self.per_query)
            for rank, chunk in enumerate(results, start=1):
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + q.weight / (config.RRF_K + rank)
                by_id.setdefault(chunk.chunk_id, chunk)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        # Collapse to the best chunk per (doc, page) when page is known.
        seen_pages: set[tuple[str, int]] = set()
        out: list[Chunk] = []
        for chunk_id, score in ranked:
            c = by_id[chunk_id]
            if c.page is not None:
                key = (c.doc_id, c.page)
                if key in seen_pages:
                    continue
                seen_pages.add(key)
            c = c.model_copy(update={"score": round(score, 5)})
            out.append(c)
            if len(out) >= self.top_k:
                break

        for i, c in enumerate(out, start=1):
            c.handle = f"c{i}"
        return out

    def search_text(self, text: str, limit: int | None = None) -> list[Chunk]:
        """Convenience for the `search` CLI command: one free-text query."""
        terms = _fts_terms(text)
        if not terms:
            return []
        results = self.store.search(_or_query(terms), limit=limit or self.top_k)
        for i, c in enumerate(results, start=1):
            c.handle = f"c{i}"
        return results
