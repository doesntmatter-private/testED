"""Capture-now, diagnose-later queue. See DESIGN.md 4.7.

Job directory layout:
  <ts>-<id>/
    prepared.json   snapshot, flags, chunks, symptoms, audio features, image paths
    request.json    the assembled API request (for inspection; rebuilt on drain)
    media/          copied audio and image files
    diagnosis.json  written on successful drain
    report.txt      rendered report, written on successful drain
    error.txt       last failure, if any
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic

from .. import config
from ..models import DiagnosisResult
from .diagnose import DiagnosisRefused, Offline, Prepared, call_model


@dataclass
class Job:
    path: Path

    @property
    def done(self) -> bool:
        return (self.path / "diagnosis.json").exists()

    @property
    def name(self) -> str:
        return self.path.name


def enqueue(p: Prepared, queue_dir: Path = config.QUEUE_DIR) -> Job:
    queue_dir.mkdir(parents=True, exist_ok=True)
    ident = p.snapshot.vehicle.vin or p.snapshot.source.split(":")[-1].replace(".json", "") or "job"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job_dir = queue_dir / f"{stamp}-{ident}"
    job_dir.mkdir(parents=True, exist_ok=False)

    media_dir = job_dir / "media"
    media_dir.mkdir()
    new_images = []
    for img in p.image_paths:
        src = Path(img)
        if src.exists():
            dst = media_dir / src.name
            shutil.copy2(src, dst)
            new_images.append(str(dst))
    if p.audio and Path(p.audio.path).exists():
        dst = media_dir / Path(p.audio.path).name
        shutil.copy2(p.audio.path, dst)
        p.audio.path = str(dst)
    p.image_paths = new_images

    (job_dir / "prepared.json").write_text(json.dumps(p.to_json(), indent=1))
    req = dict(p.request)
    # Strip base64 image payloads from the inspection copy; they are rebuilt from media/ on drain.
    req["messages"] = "<rebuilt from prepared.json on drain>"
    (job_dir / "request.json").write_text(json.dumps(req, indent=1))
    return Job(job_dir)


def list_jobs(queue_dir: Path = config.QUEUE_DIR) -> list[Job]:
    if not queue_dir.exists():
        return []
    return [Job(p) for p in sorted(queue_dir.iterdir()) if (p / "prepared.json").exists()]


def load_prepared(job: Job) -> Prepared:
    return Prepared.from_json(json.loads((job.path / "prepared.json").read_text()))


def drain(
    queue_dir: Path = config.QUEUE_DIR,
    client: anthropic.Anthropic | None = None,
    render=None,
    backend=None,
) -> list[tuple[Job, DiagnosisResult | None, str | None]]:
    """Send every pending job. Returns (job, result_or_None, error_or_None) per job.

    Failures leave the job in place for the next drain. A refusal is recorded
    but the job is also left in place so the user can inspect it.
    """
    if backend is None:
        from .backends import ClaudeBackend, make_backend

        backend = ClaudeBackend(client=client) if client is not None else make_backend()
    out = []
    for job in list_jobs(queue_dir):
        if job.done:
            continue
        try:
            p = load_prepared(job)
            result = call_model(p, backend=backend)
        except DiagnosisRefused as e:
            (job.path / "error.txt").write_text(f"refused: {e}\n")
            out.append((job, None, str(e)))
            continue
        except (Offline, RuntimeError, anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError) as e:
            (job.path / "error.txt").write_text(f"{type(e).__name__}: {e}\n")
            out.append((job, None, f"{type(e).__name__}: {e}"))
            continue
        (job.path / "diagnosis.json").write_text(result.model_dump_json(indent=1))
        if render is not None:
            (job.path / "report.txt").write_text(render(result))
        err = job.path / "error.txt"
        if err.exists():
            err.unlink()
        out.append((job, result, None))
    return out
