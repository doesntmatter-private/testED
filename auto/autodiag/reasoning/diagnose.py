"""The single structured-output call, plus the shared preparation step.

`prepare()` does everything up to the model call (flags, retrieval, prompt)
and is reused by the offline report and the queue. `call_model()` sends the
prepared context through a Backend (Claude by default, Ollama for a local
model) and validates the JSON that comes back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pydantic

from .. import config
from ..knowledge import KnowledgeStore, Retriever, build_queries
from ..models import (
    AudioFeatures,
    Chunk,
    Diagnosis,
    DiagnosisResult,
    Flag,
    VehicleSnapshot,
)
from .backends import (
    Backend,
    BackendRefused,
    BackendUnavailable,
    ClaudeBackend,
    OllamaBackend,
    make_backend,
)
from .flags import compute_flags
from .prompts import SYSTEM_PROMPT, build_user_content
from .validate import validate_diagnosis

FALLBACK_BETA = "server-side-fallback-2026-07-01"  # re-exported for callers/tests


class DiagnosisRefused(RuntimeError):
    pass


class Offline(ConnectionError):
    pass


@dataclass
class Prepared:
    snapshot: VehicleSnapshot
    flags: list[Flag]
    chunks: list[Chunk]
    symptoms: str
    audio: AudioFeatures | None
    image_paths: list[str]
    request: dict[str, Any] = field(default_factory=dict)

    def user_blocks(self) -> list[dict[str, Any]]:
        return build_user_content(self.snapshot, self.flags, self.chunks, self.symptoms, self.audio, self.image_paths)

    def to_json(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot.model_dump(),
            "flags": [f.model_dump() for f in self.flags],
            "chunks": [c.model_dump() for c in self.chunks],
            "symptoms": self.symptoms,
            "audio": self.audio.model_dump() if self.audio else None,
            "image_paths": self.image_paths,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Prepared":
        p = cls(
            snapshot=VehicleSnapshot.model_validate(d["snapshot"]),
            flags=[Flag.model_validate(f) for f in d["flags"]],
            chunks=[Chunk.model_validate(c) for c in d["chunks"]],
            symptoms=d.get("symptoms", ""),
            audio=AudioFeatures.model_validate(d["audio"]) if d.get("audio") else None,
            image_paths=d.get("image_paths", []),
        )
        p.request = build_request(p)
        return p


def output_schema() -> dict[str, Any]:
    return Diagnosis.model_json_schema()


def build_request(p: Prepared, backend: Backend | None = None) -> dict[str, Any]:
    """The exact request the backend would send. Stored in queued jobs for inspection."""
    backend = backend or make_backend()
    return backend.build_request(SYSTEM_PROMPT, p.user_blocks(), output_schema(), config.MAX_TOKENS)  # type: ignore[attr-defined]


def prepare(
    snapshot: VehicleSnapshot,
    store: KnowledgeStore,
    symptoms: str = "",
    audio: AudioFeatures | None = None,
    image_paths: list[str] | None = None,
    redact_vin: bool = False,
) -> Prepared:
    if redact_vin:
        snapshot = snapshot.redacted()
    flags = compute_flags(snapshot)
    queries = build_queries(snapshot, symptoms, flags)
    chunks = Retriever(store).search(queries)
    p = Prepared(snapshot, flags, chunks, symptoms, audio, image_paths or [])
    p.request = build_request(p)
    return p


def is_online(timeout: float = config.CONNECT_PROBE_TIMEOUT, backend: Backend | None = None) -> bool:
    """One short TCP probe to the backend's host. No retries."""
    return (backend or make_backend()).is_available(timeout)


def _parse(text: str) -> Diagnosis:
    return Diagnosis.model_validate(json.loads(text))


def call_model(
    p: Prepared,
    client=None,
    strict: bool = False,
    backend: Backend | None = None,
) -> DiagnosisResult:
    """Send the prepared context and return a validated DiagnosisResult.

    `client` is kept for backward compatibility: when given without a
    backend, it is wrapped in a ClaudeBackend.
    """
    if backend is None:
        backend = ClaudeBackend(client=client) if client is not None else make_backend()

    system = SYSTEM_PROMPT
    blocks = p.user_blocks()
    schema = output_schema()

    try:
        resp = backend.complete(system, blocks, schema, config.MAX_TOKENS)
    except BackendRefused as e:
        raise DiagnosisRefused(str(e)) from e
    except BackendUnavailable as e:
        raise Offline(str(e)) from e

    if resp.truncated:
        resp = backend.complete(system, blocks, schema, min(config.MAX_TOKENS * 4, 64000))
        if resp.truncated:
            raise RuntimeError("Model output exceeded max_tokens twice; response truncated.")

    try:
        diag = _parse(resp.text)
    except (json.JSONDecodeError, pydantic.ValidationError) as first_err:
        # Local models sometimes miss the schema on the first try. Feed the
        # error back once; Claude with output_config never reaches this path.
        if backend.name == "claude":
            raise
        repair_blocks = blocks + [
            {
                "type": "text",
                "text": (
                    "Your previous answer was not valid for the required schema. Error:\n"
                    f"{str(first_err)[:1500]}\n\nPrevious answer (truncated):\n{resp.text[:3000]}\n\n"
                    "Return only a corrected JSON object that satisfies the schema."
                ),
            }
        ]
        resp = backend.complete(system, repair_blocks, schema, config.MAX_TOKENS)
        diag = _parse(resp.text)

    report = validate_diagnosis(diag, p.chunks, strict=strict)
    return DiagnosisResult(
        diagnosis=diag,
        chunks=p.chunks,
        flags=p.flags,
        validation=report,
        model=resp.model,
        backend=backend.name,
        usage=resp.usage,
        fallback_model=resp.fallback_model,
    )


def diagnose(
    snapshot: VehicleSnapshot,
    store: KnowledgeStore,
    symptoms: str = "",
    audio: AudioFeatures | None = None,
    image_paths: list[str] | None = None,
    redact_vin: bool = False,
    client=None,
    strict: bool = False,
    backend: Backend | None = None,
) -> DiagnosisResult:
    """Full pipeline. Raises Offline if the backend host is unreachable."""
    p = prepare(snapshot, store, symptoms, audio, image_paths, redact_vin)
    backend = backend or (ClaudeBackend(client=client) if client is not None else make_backend())
    if not is_online(backend=backend):
        raise Offline(f"{backend.name} backend is unreachable")
    return call_model(p, strict=strict, backend=backend)


def choose_backend(requested: str | None = None) -> tuple[Backend, str | None]:
    """Pick a backend for the CLI.

    Returns (backend, note). `note` explains an automatic switch, or is None.
    Order: explicit request; else configured default; if that is Claude and
    unreachable and LOCAL_FALLBACK is on and Ollama answers, use Ollama.
    """
    name = requested or config.BACKEND
    primary = make_backend(name)
    if primary.is_available():
        return primary, None
    if primary.name == "claude" and config.LOCAL_FALLBACK:
        local = OllamaBackend()
        if local.is_available():
            return local, f"Claude host unreachable; using local model {local.model} via Ollama."
    return primary, None
