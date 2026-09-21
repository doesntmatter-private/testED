"""Model backends.

The pipeline builds one system prompt, one user turn (text plus optional
images), and one JSON schema. A backend turns those into a JSON string.

  ClaudeBackend  Anthropic API via the official SDK (default).
  OllamaBackend  A local model served by Ollama over HTTP. Uses Ollama's
                 `format` parameter for schema-constrained output. No extra
                 Python dependency; talks to http://localhost:11434 by default.

Local models are weaker at citation discipline and schema adherence than
Claude, so the caller validates the JSON and the report labels the backend.
"""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from .. import config


class BackendRefused(RuntimeError):
    """The model declined the request (Claude safety refusal)."""


class BackendUnavailable(ConnectionError):
    """The backend host is unreachable."""


@dataclass
class BackendResponse:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    fallback_model: str | None = None
    truncated: bool = False


class Backend(Protocol):
    name: str

    def is_available(self, timeout: float = config.CONNECT_PROBE_TIMEOUT) -> bool: ...

    def complete(
        self,
        system: str,
        user_blocks: list[dict[str, Any]],
        schema: dict[str, Any],
        max_tokens: int,
    ) -> BackendResponse: ...


def _tcp_probe(url: str, timeout: float) -> bool:
    u = urlparse(url)
    host = u.hostname or "localhost"
    port = u.port or (80 if u.scheme == "http" else 443)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeBackend:
    name = "claude"

    def __init__(self, client=None, model: str | None = None):
        self._client = client
        self.model = model or config.MODEL

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def is_available(self, timeout: float = config.CONNECT_PROBE_TIMEOUT) -> bool:
        return _tcp_probe(config.API_BASE_URL, timeout)

    def build_request(self, system: str, user_blocks: list[dict[str, Any]], schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_blocks}],
            "output_config": {"effort": config.EFFORT, "format": {"type": "json_schema", "schema": schema}},
        }
        if config.FALLBACKS_ENABLED:
            req["betas"] = [FALLBACK_BETA]
            req["fallbacks"] = "default"
        return req

    def complete(self, system, user_blocks, schema, max_tokens) -> BackendResponse:
        req = self.build_request(system, user_blocks, schema, max_tokens)
        response = self.client.beta.messages.create(**req)

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            why = getattr(details, "explanation", None) or getattr(details, "category", None) or "unspecified"
            raise BackendRefused(f"The model declined this request ({why}).")

        text = next((b.text for b in response.content if b.type == "text"), "")
        fallback_model = None
        for b in response.content:
            if b.type == "fallback":
                fallback_model = getattr(getattr(b, "to", None), "model", None)
        usage = response.usage
        return BackendResponse(
            text=text,
            model=response.model,
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            },
            fallback_model=fallback_model,
            truncated=response.stop_reason == "max_tokens",
        )


# --------------------------------------------------------------------------- #
# Ollama (local)
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


class OllamaBackend:
    """POST {host}/api/chat with `format` set to the JSON schema.

    Images in the user blocks are forwarded as base64 in the message's
    `images` list, which vision-capable local models (llava, qwen-vl,
    gemma3) accept; text-only models ignore them.
    """

    name = "ollama"

    def __init__(self, host: str | None = None, model: str | None = None, timeout: float | None = None,
                 opener=None):
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.model = model or config.LOCAL_MODEL
        self.timeout = timeout or config.LOCAL_TIMEOUT
        self._open = opener or urllib.request.urlopen

    def is_available(self, timeout: float = config.CONNECT_PROBE_TIMEOUT) -> bool:
        return _tcp_probe(self.host, timeout)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:500]
            if e.code == 404 and "not found" in body.lower():
                raise RuntimeError(
                    f"Ollama model {self.model!r} is not pulled. Run: ollama pull {self.model}"
                ) from e
            raise RuntimeError(f"Ollama HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, OSError) as e:
            raise BackendUnavailable(f"Ollama at {self.host} is unreachable: {e}") from e

    def build_request(self, system: str, user_blocks: list[dict[str, Any]], schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        texts: list[str] = []
        images: list[str] = []
        for b in user_blocks:
            if b.get("type") == "text":
                texts.append(b["text"])
            elif b.get("type") == "image":
                images.append(b["source"]["data"])
        user_msg: dict[str, Any] = {"role": "user", "content": "\n\n".join(texts)}
        if images:
            user_msg["images"] = images
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, user_msg],
            "format": schema,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": config.LOCAL_NUM_CTX,
                "temperature": 0.2,
            },
        }

    def complete(self, system, user_blocks, schema, max_tokens) -> BackendResponse:
        payload = self.build_request(system, user_blocks, schema, max_tokens)
        data = self._post("/api/chat", payload)
        text = _strip_fences(data.get("message", {}).get("content", ""))
        return BackendResponse(
            text=text,
            model=data.get("model", self.model),
            usage={
                "input_tokens": int(data.get("prompt_eval_count", 0) or 0),
                "output_tokens": int(data.get("eval_count", 0) or 0),
            },
            truncated=data.get("done_reason") == "length",
        )


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def make_backend(name: str | None = None, **kwargs) -> Backend:
    name = (name or config.BACKEND).lower()
    if name == "claude":
        return ClaudeBackend(**kwargs)
    if name == "ollama":
        return OllamaBackend(**kwargs)
    raise ValueError(f"Unknown backend {name!r}; expected 'claude' or 'ollama'")
