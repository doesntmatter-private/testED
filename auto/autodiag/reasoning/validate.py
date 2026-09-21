"""Post-generation checks: citation validity, probability normalization, cause references."""

from __future__ import annotations

import re

from ..models import Chunk, Diagnosis, ValidationReport

HANDLE_RE = re.compile(r"^c\d+$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def validate_diagnosis(diag: Diagnosis, chunks: list[Chunk], strict: bool = False) -> ValidationReport:
    """Mutates `diag` in place: downgrades bad manual citations, renormalizes probabilities."""
    handles = {c.handle: c for c in chunks}
    report = ValidationReport()

    for cause in diag.causes:
        for ev in cause.evidence:
            if ev.kind != "manual":
                continue
            src = ev.source.strip()
            chunk = handles.get(src)
            bad = chunk is None
            if not bad and strict and chunk is not None:
                bad = _norm(ev.detail)[:60] not in _norm(chunk.text) and not _quoted_fragment_in(ev.detail, chunk.text)
            if bad:
                report.invalid_citations += 1
                report.downgraded.append(f"{cause.title}: '{src or '<empty>'}'")
                ev.kind = "general_knowledge"
                ev.source = ""

    for test in diag.next_tests:
        src = test.source.strip()
        if src and (HANDLE_RE.match(src) and src not in handles):
            report.invalid_citations += 1
            report.downgraded.append(f"test '{test.name}': '{src}'")
            test.source = ""

    cause_titles = {_norm(c.title) for c in diag.causes}
    for test in diag.next_tests:
        for ref in test.discriminates:
            if _norm(ref) not in cause_titles and not any(_norm(ref) in t or t in _norm(ref) for t in cause_titles):
                report.unknown_cause_refs.append(f"{test.name} -> {ref}")

    total = sum(c.probability for c in diag.causes)
    report.probability_sum_before = round(total, 3)
    if total > 0 and abs(total - 1.0) > 0.02:
        for c in diag.causes:
            c.probability = round(c.probability / total, 3)

    diag.causes.sort(key=lambda c: c.probability, reverse=True)
    for i, c in enumerate(diag.causes, start=1):
        c.rank = i
    diag.next_tests.sort(key=lambda t: t.order)
    for i, t in enumerate(diag.next_tests, start=1):
        t.order = i
    return report


def _quoted_fragment_in(detail: str, text: str) -> bool:
    """True if any quoted fragment in `detail` appears in `text`."""
    for frag in re.findall(r'"([^"]{8,})"|“([^”]{8,})”', detail):
        f = frag[0] or frag[1]
        if _norm(f) in _norm(text):
            return True
    return False
