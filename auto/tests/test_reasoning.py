import json

from autodiag.models import Cause, Chunk, Diagnosis, Evidence, NextTest
from autodiag.reasoning.diagnose import Prepared, build_request, output_schema, prepare
from autodiag.reasoning.flags import compute_flags
from autodiag.reasoning.offline import render_offline_report
from autodiag.reasoning.validate import validate_diagnosis


def _chunks():
    return [
        Chunk(chunk_id="d:0", doc_id="d", title="Misfire", page=None, section="Swap", text="Swap the coil with a known-good cylinder.", handle="c1"),
        Chunk(chunk_id="d:1", doc_id="d", title="Misfire", page=None, section="Plug", text="Inspect the spark plug.", handle="c2"),
    ]


def _diag():
    return Diagnosis(
        summary="s",
        severity="limit_driving",
        causes=[
            Cause(rank=1, title="Coil", probability=0.5, reasoning="r",
                  evidence=[Evidence(kind="manual", detail="Swap the coil", source="c1"),
                            Evidence(kind="manual", detail="made up", source="c9"),
                            Evidence(kind="obd", detail="P0301", source="P0301")],
                  contradicting_evidence=[]),
            Cause(rank=2, title="Plug", probability=0.3, reasoning="r",
                  evidence=[Evidence(kind="general_knowledge", detail="plugs wear", source="")],
                  contradicting_evidence=[]),
        ],
        next_tests=[
            NextTest(order=2, name="Plug", procedure="p", discriminates=["Plug"], expected_if_faulty="a", expected_if_ok="b", tools_needed=[], est_minutes=5, source="c2"),
            NextTest(order=1, name="Swap", procedure="p", discriminates=["Coil", "Nonexistent cause"], expected_if_faulty="a", expected_if_ok="b", tools_needed=[], est_minutes=20, source="c7"),
        ],
        missing_information=[],
        safety_notes=[],
    )


def test_validate_downgrades_bad_citations_and_normalizes():
    d = _diag()
    rep = validate_diagnosis(d, _chunks())
    assert rep.invalid_citations == 2  # c9 evidence + c7 test source
    bad = d.causes[0].evidence[1]
    assert bad.kind == "general_knowledge" and bad.source == ""
    assert d.causes[0].evidence[0].kind == "manual"  # c1 kept
    assert rep.probability_sum_before == 0.8
    assert abs(sum(c.probability for c in d.causes) - 1.0) < 0.01
    assert [t.order for t in d.next_tests] == [1, 2] and d.next_tests[0].name == "Swap"
    assert d.next_tests[0].source == ""
    assert rep.unknown_cause_refs == ["Swap -> Nonexistent cause"]


def test_validate_strict_requires_text_match():
    d = _diag()
    d.causes[0].evidence[0].detail = "Something not in the chunk at all"
    rep = validate_diagnosis(d, _chunks(), strict=True)
    assert rep.invalid_citations == 3


def test_flags_for_lean_fixture(lean_snapshot):
    names = {f.name for f in compute_flags(lean_snapshot)}
    assert "lean_bank1_freeze_frame" in names
    assert "map_high_idle_freeze_frame" in names
    assert "lean_bank1_live" not in names  # trims normal at 2500 rpm


def test_flags_for_misfire_fixture(misfire_snapshot):
    flags = compute_flags(misfire_snapshot)
    names = {f.name for f in flags}
    assert "single_cylinder_misfire" in names
    assert "map_normal_idle_live" in names
    assert not any(n.startswith("lean") for n in names)


def test_output_schema_is_strict():
    schema = output_schema()
    assert schema["additionalProperties"] is False
    for name, sub in schema["$defs"].items():
        assert sub.get("additionalProperties") is False, name
    assert "causes" in schema["required"]


def test_build_request_shape(store, misfire_snapshot):
    p = prepare(misfire_snapshot, store, "rough idle")
    req = p.request
    assert req["model"]
    assert req["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["betas"]
    assert req["output_config"]["format"]["type"] == "json_schema"
    text = req["messages"][0]["content"][0]["text"]
    assert "[c1]" in text and "P0301" in text and "rough idle" in text
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_build_request_gateway_mode(store, misfire_snapshot, monkeypatch):
    from autodiag import config

    monkeypatch.setattr(config, "FALLBACKS_ENABLED", False)
    monkeypatch.setattr(config, "MODEL", "azure/anthropic/claude-opus-5")
    req = build_request(prepare(misfire_snapshot, store))
    assert req["model"] == "azure/anthropic/claude-opus-5"
    assert "betas" not in req and "fallbacks" not in req
    assert req["output_config"]["format"]["type"] == "json_schema"


def test_is_online_follows_base_url(monkeypatch):
    import importlib

    from autodiag import config

    d = importlib.import_module("autodiag.reasoning.diagnose")
    b = importlib.import_module("autodiag.reasoning.backends")

    seen = {}

    class FakeSock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_connect(addr, timeout):
        seen["addr"] = addr
        return FakeSock()

    monkeypatch.setattr(config, "API_BASE_URL", "https://gateway.example.com:8443/v1")
    monkeypatch.setattr(b.socket, "create_connection", fake_connect)
    assert d.is_online() is True
    assert seen["addr"] == ("gateway.example.com", 8443)


def test_prepared_roundtrip(store, misfire_snapshot):
    p = prepare(misfire_snapshot, store, "rough idle")
    p2 = Prepared.from_json(json.loads(json.dumps(p.to_json())))
    assert [c.handle for c in p2.chunks] == [c.handle for c in p.chunks]
    assert p2.request["messages"] == p.request["messages"]


def test_offline_report_contents(store, misfire_snapshot):
    p = prepare(misfire_snapshot, store, "")
    text = render_offline_report(misfire_snapshot, p.flags, p.chunks)
    assert "OFFLINE REPORT" in text
    assert "P0301" in text and "Cylinder 1 Misfire" in text
    assert "-- P0301 --" in text
    assert "MISFIRE_MONITORING: complete" in text
    assert "[c1]" in text


def test_image_blocks_only_supported_types(tmp_path):
    from autodiag.reasoning.prompts import image_blocks

    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "b.txt").write_text("x")
    blocks = image_blocks([str(img), str(tmp_path / "b.txt")])
    assert len(blocks) == 2 and blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/png"
