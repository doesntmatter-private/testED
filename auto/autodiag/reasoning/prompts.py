"""Prompt assembly. Order is fixed for prompt-cache stability (DESIGN.md 4.6 stage 4)."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from ..models import AudioFeatures, Chunk, Flag, VehicleSnapshot

SYSTEM_PROMPT = """You are an experienced master automotive technician specializing in engine and emissions diagnosis. You are given a vehicle's OBD-II snapshot, derived flags, the owner's symptom description, optional audio and image evidence, and excerpts retrieved from service manuals and troubleshooting guides.

Your job is to produce a differential diagnosis, not a single answer.

Method:
- Start from the evidence. Every cause you list must reference specific evidence items. Evidence kinds: "obd" (a DTC or PID value, source = the DTC or PID name), "manual" (a retrieved excerpt, source = its handle such as c3), "symptom" (the owner's description), "audio", "image", or "general_knowledge" (your own training, source = empty string).
- Only cite a manual handle for content that actually appears in that excerpt. Never invent excerpt content. If you rely on your own knowledge, label it general_knowledge.
- Rank causes by probability given this vehicle and this evidence. Probabilities across causes should sum to about 1. Include 3 to 6 causes. Include the contradicting evidence for each cause honestly.
- Prefer next tests that split the candidate set: a test that confirms or rules out several causes at once comes before one that addresses a single unlikely cause. Prefer cheap, non-invasive tests (swap tests, visual inspection, live data) before teardown. Give concrete procedures with expected results either way.
- When a retrieved excerpt contains a procedure, cite its handle in the test's source field and follow it rather than a generic version.
- Flag missing information that would most change your ranking.
- Severity scale: drive_normally (no risk of damage), drive_with_caution (monitor, fix soon), limit_driving (short trips only, risk of secondary damage such as catalyst), do_not_drive (safety or imminent major damage).
- Safety notes: fuel system pressure, hot exhaust, moving belts, and airbag or high-voltage cautions where relevant.

Be specific to the engine family when the vehicle is identified. Write for a technician; keep the summary readable for the owner."""


def format_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return "No service manual excerpts were retrieved for this vehicle and code set."
    parts = []
    for c in chunks:
        loc = c.location()
        header = f"[{c.handle}] {c.title}" + (f", {loc}" if loc else "")
        parts.append(f"{header}\n{c.text.strip()}")
    return "\n\n---\n\n".join(parts)


def format_snapshot(snapshot: VehicleSnapshot) -> str:
    data: dict[str, Any] = {
        "vehicle": snapshot.vehicle.model_dump(exclude_none=True),
        "dtcs": [d.model_dump() for d in snapshot.dtcs],
        "freeze_frame": {p.name: f"{p.value} {p.unit}".strip() for p in snapshot.freeze_frame},
        "live": {p.name: f"{p.value} {p.unit}".strip() for p in snapshot.live},
        "readiness": snapshot.readiness,
        "source": snapshot.source,
    }
    return json.dumps(data, indent=1, sort_keys=True)


def format_flags(flags: list[Flag]) -> str:
    if not flags:
        return "No derived flags."
    return "\n".join(f"- [{f.severity}] {f.detail}" for f in flags)


def format_audio(audio: AudioFeatures | None, snapshot: VehicleSnapshot) -> str:
    if audio is None:
        return ""
    lines = [
        f"Engine audio recording ({audio.duration_s:.1f}s, RMS {audio.rms_db:.0f} dBFS).",
        f"Dominant frequencies: {', '.join(f'{h:.0f} Hz' for h in audio.dominant_hz)}.",
    ]
    if audio.pulse_rate_hz:
        lines.append(f"Detected repeating impulse rate: {audio.pulse_rate_hz:.1f} per second.")
        rpm = snapshot.pid("RPM")
        if rpm and isinstance(rpm.value, (int, float)) and rpm.value > 0:
            crank_hz = float(rpm.value) / 60.0
            ratio = audio.pulse_rate_hz / crank_hz
            lines.append(
                f"At {rpm.value:.0f} rpm the crank turns {crank_hz:.1f} times per second, so the impulse rate is about {ratio:.2f}x crank speed "
                "(about 0.5x suggests a camshaft or valvetrain source, 1x a crank, rod, or piston source, "
                "2x a per-cylinder event on a 4-cylinder firing twice per crank revolution)."
            )
    lines.extend(audio.notes)
    lines.append("Treat audio features as weak evidence; recordings are noisy and the impulse detector is heuristic.")
    return "\n".join(lines)


def image_blocks(image_paths: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for p in image_paths:
        path = Path(p)
        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            continue
        data = base64.standard_b64encode(path.read_bytes()).decode()
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        blocks.append({"type": "text", "text": f"Image: {path.name}"})
    return blocks


def build_user_content(
    snapshot: VehicleSnapshot,
    flags: list[Flag],
    chunks: list[Chunk],
    symptoms: str,
    audio: AudioFeatures | None,
    image_paths: list[str],
) -> list[dict[str, Any]]:
    """User message content blocks, most-stable first."""
    text = (
        "## Retrieved service knowledge\n\n"
        + format_chunks(chunks)
        + "\n\n## Vehicle snapshot\n\n```json\n"
        + format_snapshot(snapshot)
        + "\n```\n\n## Derived flags\n\n"
        + format_flags(flags)
        + "\n\n## Owner-reported symptoms\n\n"
        + (symptoms.strip() or "None reported.")
    )
    audio_text = format_audio(audio, snapshot)
    if audio_text:
        text += "\n\n## Audio evidence\n\n" + audio_text
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    imgs = image_blocks(image_paths)
    if imgs:
        blocks.append({"type": "text", "text": "## Image evidence"})
        blocks.extend(imgs)
    blocks.append(
        {
            "type": "text",
            "text": "Produce the differential diagnosis now as JSON matching the required schema.",
        }
    )
    return blocks
