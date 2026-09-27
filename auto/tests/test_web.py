import io
import json

import pytest
from fastapi.testclient import TestClient

from autodiag import config
from autodiag.knowledge import KnowledgeStore, ingest_path
from autodiag.reasoning import diagnose as _d
from autodiag.reasoning.backends import OllamaBackend
from autodiag.web.app import create_app
from tests.test_backends import FakeHTTP, _ollama_reply
from tests.test_queue import _fake_diagnosis_json


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOME", tmp_path)
    monkeypatch.setattr(config, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    store = KnowledgeStore(":memory:")
    ingest_path(store, config.SEED_KNOWLEDGE_DIR)
    app = create_app(store=store)
    with TestClient(app) as c:
        yield c


def _snapshot(client):
    return client.post("/api/scan", json={"fixture": "p0301_misfire.json"}).json()


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "autodiag" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_health_and_fixtures(client, monkeypatch):
    monkeypatch.setattr("autodiag.reasoning.backends._tcp_probe", lambda url, timeout: False)
    h = client.get("/api/health").json()
    assert h["knowledge_chunks"] > 0
    assert h["backends"]["claude"]["reachable"] is False
    assert h["queue_pending"] == 0
    fx = client.get("/api/fixtures").json()
    names = {f["name"] for f in fx}
    assert {"p0301_misfire.json", "p0171_lean.json"} <= names
    assert any("Honda" in f["label"] and "P0301" in f["label"] for f in fx)


def test_scan_fixture_and_missing(client):
    snap = _snapshot(client)
    assert snap["dtcs"][0]["code"] == "P0301"
    assert client.post("/api/scan", json={"fixture": "nope.json"}).status_code == 404
    assert client.post("/api/scan", json={"fixture": "../pyproject.toml"}).status_code == 404


def test_diagnose_offline(client):
    snap = _snapshot(client)
    r = client.post("/api/diagnose", data={"snapshot": json.dumps(snap), "symptoms": "rough idle", "offline": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "offline"
    assert "OFFLINE REPORT" in body["report"] and "P0301" in body["report"]
    assert body["chunks"][0]["handle"] == "c1"


def test_diagnose_invalid_snapshot(client):
    r = client.post("/api/diagnose", data={"snapshot": "{bad", "offline": "true"})
    assert r.status_code == 400


def test_diagnose_backend_unreachable_falls_back(client, monkeypatch):
    monkeypatch.setattr("autodiag.reasoning.backends._tcp_probe", lambda url, timeout: False)
    snap = _snapshot(client)
    r = client.post("/api/diagnose", data={"snapshot": json.dumps(snap)})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "offline" and "unreachable" in body["note"]


def test_diagnose_with_fake_ollama_and_image(client, monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr("autodiag.reasoning.backends._tcp_probe", lambda url, timeout: True)
    snap = _snapshot(client)
    # Handles are assigned in rank order, so the top chunk is always c1.
    http = FakeHTTP([_ollama_reply(_fake_diagnosis_json([SimpleNamespace(handle="c1")]))])
    fake = OllamaBackend(host="http://fake:11434", model="fake", opener=http)
    monkeypatch.setattr("autodiag.web.app.choose_backend", lambda name=None: (fake, None))

    r = client.post(
        "/api/diagnose",
        data={"snapshot": json.dumps(snap), "symptoms": "rough idle", "backend": "ollama"},
        files=[("images", ("plug.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png"))],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "diagnosis"
    assert body["result"]["backend"] == "ollama"
    assert body["result"]["diagnosis"]["causes"][0]["title"].startswith("Cylinder 1")
    assert http.requests[0]["messages"][1]["images"]  # image reached the backend
    assert body["report_name"].startswith("diagnosis-")

    reports = client.get("/api/reports").json()
    assert reports[0]["name"] == body["report_name"]
    full = client.get(f"/api/reports/{body['report_name']}").json()
    assert full["diagnosis"]["severity"] == "limit_driving"
    assert client.get("/api/reports/does-not-exist").status_code == 404


def test_queue_roundtrip(client):
    snap = _snapshot(client)
    r = client.post("/api/queue", data={"snapshot": json.dumps(snap), "symptoms": "a"})
    assert r.status_code == 200
    jobs = client.get("/api/queue").json()
    assert len(jobs) == 1 and jobs[0]["status"] == "pending"
    assert client.get("/api/health").json()["queue_pending"] == 1


def test_knowledge_upload_and_search(client):
    before = client.get("/api/health").json()["knowledge_chunks"]
    r = client.post("/api/knowledge", files=[("files", ("tsb.md", io.BytesIO(b"# TSB 42\n\n## Fix\n\nP0999 flux capacitor misalignment; realign and clear codes.\n"), "text/markdown"))])
    assert r.status_code == 200
    assert r.json()["total_chunks"] == before + 1
    docs = client.get("/api/knowledge").json()
    assert any(d["title"] == "TSB 42" for d in docs)
    hits = client.get("/api/search", params={"q_": "flux capacitor"}).json()
    assert hits and "flux" in hits[0]["text"].lower()
