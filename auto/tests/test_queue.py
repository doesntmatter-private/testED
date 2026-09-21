import json
from types import SimpleNamespace

import anthropic
import pytest

from autodiag.models import Diagnosis
from autodiag.reasoning import queue as q
from autodiag.reasoning.diagnose import DiagnosisRefused, call_model, prepare


def _fake_diagnosis_json(chunks):
    return json.dumps(
        {
            "summary": "Likely a coil.",
            "severity": "limit_driving",
            "causes": [
                {
                    "rank": 1, "title": "Cylinder 1 ignition coil failure", "probability": 0.6, "reasoning": "r",
                    "evidence": [{"kind": "manual", "detail": "swap test", "source": chunks[0].handle},
                                 {"kind": "obd", "detail": "P0301 stored", "source": "P0301"}],
                    "contradicting_evidence": [],
                },
                {
                    "rank": 2, "title": "Spark plug", "probability": 0.4, "reasoning": "r",
                    "evidence": [{"kind": "general_knowledge", "detail": "plugs wear", "source": ""}],
                    "contradicting_evidence": [],
                },
            ],
            "next_tests": [
                {"order": 1, "name": "Coil swap", "procedure": "swap", "discriminates": ["Cylinder 1 ignition coil failure"],
                 "expected_if_faulty": "moves", "expected_if_ok": "stays", "tools_needed": [], "est_minutes": 30, "source": chunks[0].handle}
            ],
            "missing_information": [],
            "safety_notes": [],
        }
    )


class FakeMessages:
    def __init__(self, text=None, stop_reason="end_turn", fail=None):
        self.text = text
        self.stop_reason = stop_reason
        self.fail = fail
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category="other", explanation="nope"),
            content=[SimpleNamespace(type="text", text=self.text)],
            model="claude-opus-5",
            usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )


def fake_client(messages: FakeMessages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def test_call_model_parses_and_validates(store, misfire_snapshot):
    p = prepare(misfire_snapshot, store, "rough idle")
    fm = FakeMessages(text=_fake_diagnosis_json(p.chunks))
    result = call_model(p, client=fake_client(fm))
    assert result.diagnosis.causes[0].title.startswith("Cylinder 1")
    assert result.validation.invalid_citations == 0
    assert result.usage["input_tokens"] == 100
    sent = fm.calls[0]
    assert sent["fallbacks"] == "default"
    assert sent["output_config"]["format"]["type"] == "json_schema"


def test_call_model_refusal(store, misfire_snapshot):
    p = prepare(misfire_snapshot, store)
    fm = FakeMessages(text="", stop_reason="refusal")
    with pytest.raises(DiagnosisRefused):
        call_model(p, client=fake_client(fm))


def test_queue_roundtrip(store, misfire_snapshot, tmp_path):
    img = tmp_path / "plug.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    p = prepare(misfire_snapshot, store, "rough idle", image_paths=[str(img)])
    qdir = tmp_path / "queue"
    job = q.enqueue(p, queue_dir=qdir)
    assert (job.path / "prepared.json").exists()
    assert (job.path / "media" / "plug.png").exists()
    assert not job.done
    assert [j.name for j in q.list_jobs(qdir)] == [job.name]

    loaded = q.load_prepared(job)
    assert loaded.image_paths == [str(job.path / "media" / "plug.png")]
    assert [c.handle for c in loaded.chunks] == [c.handle for c in p.chunks]


def test_drain_success_and_failure(store, misfire_snapshot, tmp_path):
    qdir = tmp_path / "queue"
    p1 = prepare(misfire_snapshot, store, "a")
    j1 = q.enqueue(p1, queue_dir=qdir)
    # Force a distinct directory name for the second job
    import time
    time.sleep(1.1)
    p2 = prepare(misfire_snapshot, store, "b")
    j2 = q.enqueue(p2, queue_dir=qdir)

    # First drain: connection error -> both jobs remain
    fm_fail = FakeMessages(fail=anthropic.APIConnectionError(request=SimpleNamespace()))
    out = q.drain(qdir, client=fake_client(fm_fail))
    assert all(r is None for _, r, _ in out)
    assert (j1.path / "error.txt").exists()
    assert not j1.done and not j2.done

    # Second drain: success -> diagnosis.json written, error cleared
    fm_ok = FakeMessages(text=_fake_diagnosis_json(p1.chunks))
    out = q.drain(qdir, client=fake_client(fm_ok), render=lambda r: "REPORT")
    assert all(r is not None for _, r, _ in out)
    assert j1.done and j2.done
    assert not (j1.path / "error.txt").exists()
    assert (j1.path / "report.txt").read_text() == "REPORT"
    Diagnosis.model_validate(json.loads((j1.path / "diagnosis.json").read_text())["diagnosis"])

    # Third drain: nothing pending
    assert q.drain(qdir, client=fake_client(FakeMessages(fail=RuntimeError("should not be called")))) == []
