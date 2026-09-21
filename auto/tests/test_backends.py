import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from autodiag import config
from autodiag.reasoning import diagnose as _dmod
from autodiag.reasoning.backends import ClaudeBackend, OllamaBackend, make_backend
from autodiag.reasoning.diagnose import Offline, call_model, choose_backend, prepare
from tests.test_queue import _fake_diagnosis_json


class FakeHTTP:
    """Stands in for urllib.request.urlopen. Returns canned Ollama responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(json.loads(req.data.decode()))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        body = json.dumps(item).encode()

        class Resp(io.BytesIO):
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        return Resp(body)


def _ollama_reply(content, done_reason="stop", model="qwen2.5:14b"):
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 4200,
        "eval_count": 900,
    }


def test_ollama_request_shape(store, misfire_snapshot, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    p = prepare(misfire_snapshot, store, "rough idle", image_paths=[str(img)])
    http = FakeHTTP([_ollama_reply(_fake_diagnosis_json(p.chunks))])
    be = OllamaBackend(host="http://ollama.local:11434", model="llama3.1:8b", opener=http)

    result = call_model(p, backend=be)

    sent = http.requests[0]
    assert sent["model"] == "llama3.1:8b"
    assert sent["stream"] is False
    assert sent["format"]["type"] == "object" and "causes" in sent["format"]["properties"]
    assert sent["messages"][0]["role"] == "system"
    assert "[c1]" in sent["messages"][1]["content"] and "P0301" in sent["messages"][1]["content"]
    assert sent["messages"][1]["images"]  # base64 image forwarded
    assert sent["options"]["num_predict"] == config.MAX_TOKENS
    assert result.backend == "ollama"
    assert result.model == "qwen2.5:14b"
    assert result.usage == {"input_tokens": 4200, "output_tokens": 900}
    assert result.validation.invalid_citations == 0


def test_ollama_strips_fences_and_repairs_bad_json(store, misfire_snapshot):
    p = prepare(misfire_snapshot, store)
    good = _fake_diagnosis_json(p.chunks)
    http = FakeHTTP([
        _ollama_reply("```json\n{\"summary\": \"incomplete\"}\n```"),  # fails schema
        _ollama_reply(good),  # repair attempt
    ])
    be = OllamaBackend(opener=http)
    result = call_model(p, backend=be)
    assert result.diagnosis.causes[0].title.startswith("Cylinder 1")
    assert len(http.requests) == 2
    assert "not valid for the required schema" in http.requests[1]["messages"][1]["content"]


def test_ollama_repair_gives_up_after_one_retry(store, misfire_snapshot):
    p = prepare(misfire_snapshot, store)
    http = FakeHTTP([_ollama_reply("not json"), _ollama_reply("still not json")])
    with pytest.raises(json.JSONDecodeError):
        call_model(p, backend=OllamaBackend(opener=http))


def test_ollama_truncation_retries_with_more_tokens(store, misfire_snapshot):
    p = prepare(misfire_snapshot, store)
    good = _fake_diagnosis_json(p.chunks)
    http = FakeHTTP([_ollama_reply(good[:50], done_reason="length"), _ollama_reply(good)])
    call_model(p, backend=OllamaBackend(opener=http))
    assert http.requests[1]["options"]["num_predict"] > http.requests[0]["options"]["num_predict"]


def test_ollama_model_not_pulled_message(store, misfire_snapshot):
    p = prepare(misfire_snapshot, store)
    err = urllib.error.HTTPError("u", 404, "Not Found", {}, io.BytesIO(b'{"error":"model \'x\' not found"}'))
    with pytest.raises(RuntimeError, match="ollama pull"):
        call_model(p, backend=OllamaBackend(model="x", opener=FakeHTTP([err])))


def test_ollama_unreachable_maps_to_offline(store, misfire_snapshot):
    p = prepare(misfire_snapshot, store)
    err = urllib.error.URLError("connection refused")
    with pytest.raises(Offline):
        call_model(p, backend=OllamaBackend(opener=FakeHTTP([err])))


def test_claude_backend_does_not_repair(store, misfire_snapshot):
    """With output_config the API guarantees valid JSON; a parse failure is a real bug, not retried."""
    p = prepare(misfire_snapshot, store)
    fake = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    stop_reason="end_turn", content=[SimpleNamespace(type="text", text="{}")],
                    model="claude-opus-5",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0),
                )
            )
        )
    )
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        call_model(p, backend=ClaudeBackend(client=fake))


def test_make_backend_selection(monkeypatch):
    assert make_backend("claude").name == "claude"
    assert make_backend("ollama").name == "ollama"
    monkeypatch.setattr(config, "BACKEND", "ollama")
    assert make_backend().name == "ollama"
    with pytest.raises(ValueError):
        make_backend("gpt")


def test_choose_backend_local_fallback(monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "claude")
    monkeypatch.setattr(config, "LOCAL_FALLBACK", True)
    monkeypatch.setattr(ClaudeBackend, "is_available", lambda self, timeout=3.0: False)
    monkeypatch.setattr(OllamaBackend, "is_available", lambda self, timeout=3.0: True)
    be, note = choose_backend()
    assert be.name == "ollama" and "local model" in note

    monkeypatch.setattr(config, "LOCAL_FALLBACK", False)
    be, note = choose_backend()
    assert be.name == "claude" and note is None

    be, note = choose_backend("ollama")
    assert be.name == "ollama" and note is None
