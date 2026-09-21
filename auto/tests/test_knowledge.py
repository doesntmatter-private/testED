from pathlib import Path

from autodiag.knowledge import KnowledgeStore, Retriever, build_queries, ingest_path
from autodiag.knowledge.ingest import chunk_text
from autodiag.reasoning.flags import compute_flags


def test_ingest_is_idempotent(tmp_path):
    doc = tmp_path / "a.md"
    doc.write_text("# Title\n\n## Sec\n\nP0301 cylinder 1 misfire coil swap test.\n")
    s = KnowledgeStore(":memory:")
    r1 = ingest_path(s, doc)
    r2 = ingest_path(s, doc)
    assert r1[0][2] == "ingested" and r1[0][1] == 1
    assert r2[0][2].startswith("skipped")
    assert s.count_chunks() == 1
    row = s.list_documents()[0]
    assert row["title"] == "Title"


def test_markdown_sections_keep_headings(store):
    hits = store.search('"swap"', limit=5)
    assert hits
    assert any(h.section and "Swap" in h.section for h in hits)


def test_chunking_respects_target_and_overlaps():
    step = "\n".join(f"{i}. Step {i} does something with the engine and the sensor readings." for i in range(1, 200))
    chunks = chunk_text(step, target=200, overlap=40)
    assert len(chunks) > 3
    # overlap: last unit of chunk i appears in chunk i+1
    for a, b in zip(chunks, chunks[1:]):
        last_line = a.split("\n")[-1]
        assert last_line in b
    # every chunk begins on a step boundary
    for c in chunks:
        assert c.lstrip()[0].isdigit()


def test_build_queries_shapes(misfire_snapshot):
    flags = compute_flags(misfire_snapshot)
    qs = build_queries(misfire_snapshot, "rough idle shake at stoplights", flags)
    kinds = [q.kind for q in qs]
    assert kinds.count("dtc") == 2
    assert "symptoms" in kinds and "vehicle" in kinds and "combined" in kinds
    dtc_q = next(q for q in qs if q.kind == "dtc")
    assert '"p0301"' in dtc_q.text or '"p0300"' in dtc_q.text
    assert dtc_q.weight == 2.0
    # stopwords removed
    sym = next(q for q in qs if q.kind == "symptoms")
    assert '"at"' not in sym.text and '"rough"' in sym.text


def test_retrieval_prefers_misfire_doc_for_p0301(store, misfire_snapshot):
    flags = compute_flags(misfire_snapshot)
    chunks = Retriever(store).search(build_queries(misfire_snapshot, "rough idle", flags))
    assert chunks
    assert chunks[0].handle == "c1"
    assert [c.handle for c in chunks] == [f"c{i}" for i in range(1, len(chunks) + 1)]
    assert "Misfire" in chunks[0].title
    # the swap test should be in the top few
    assert any("swap" in c.text.lower() for c in chunks[:4])


def test_retrieval_prefers_lean_doc_for_p0171(store, lean_snapshot):
    flags = compute_flags(lean_snapshot)
    chunks = Retriever(store).search(build_queries(lean_snapshot, "", flags))
    assert "Lean" in chunks[0].title


def test_search_bad_expression_does_not_crash(store):
    assert store.search('"unterminated', limit=3) == []
    assert store.search("", limit=3) == []


def test_ingest_directory_and_pdf_skip(tmp_path):
    (tmp_path / "x.txt").write_text("plain text about P0420 catalyst efficiency")
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01")
    s = KnowledgeStore(":memory:")
    res = ingest_path(s, tmp_path)
    assert [r[0] for r in res] == ["x.txt"]
    assert Path(s.list_documents()[0]["path"]).name == "x.txt"
