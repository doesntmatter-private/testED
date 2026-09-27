"""FastAPI application.

Every endpoint is a thin wrapper over functions in autodiag.reasoning,
autodiag.knowledge, and autodiag.obd. No diagnostic logic lives here.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import anthropic
import pydantic
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .. import config
from ..knowledge import KnowledgeStore, Retriever, ingest_path
from ..models import DiagnosisResult, VehicleSnapshot
from ..obd import open_reader
from ..reasoning import queue as q
from ..reasoning.backends import ClaudeBackend, OllamaBackend, make_backend
from ..reasoning.diagnose import DiagnosisRefused, Offline, Prepared, call_model, choose_backend, prepare
from ..reasoning.offline import render_offline_report

STATIC_DIR = Path(__file__).parent / "static"


def create_app(store: KnowledgeStore | None = None) -> FastAPI:
    app = FastAPI(title="autodiag", version="0.1.0", docs_url="/api/docs", redoc_url=None)
    config.ensure_dirs()
    app.state.store = store or KnowledgeStore(config.KB_PATH)
    app.state.upload_dir = Path(tempfile.mkdtemp(prefix="autodiag_web_"))

    # ---- helpers ---------------------------------------------------------

    def _store() -> KnowledgeStore:
        return app.state.store

    async def _save_upload(f: UploadFile) -> str:
        dst = app.state.upload_dir / f"{int(time.time() * 1000)}_{Path(f.filename or 'upload').name}"
        with dst.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        return str(dst)

    async def _prepare_from_form(
        snapshot: str,
        symptoms: str,
        no_vin: bool,
        audio: UploadFile | None,
        images: list[UploadFile],
    ) -> tuple[VehicleSnapshot, Prepared]:
        try:
            snap = VehicleSnapshot.model_validate_json(snapshot)
        except pydantic.ValidationError as e:
            raise HTTPException(400, f"Invalid snapshot: {e.errors()[0]['msg']}")
        audio_feats = None
        if audio and audio.filename:
            from ..media import analyze_audio

            path = await _save_upload(audio)
            try:
                audio_feats = analyze_audio(path)
            except Exception as e:  # noqa: BLE001 - surface any decode problem to the user
                raise HTTPException(400, f"Could not analyze audio: {e}")
        image_paths = [await _save_upload(img) for img in images if img.filename]
        p = prepare(snap, _store(), symptoms, audio_feats, image_paths, redact_vin=no_vin)
        return snap, p

    def _save_report(result: DiagnosisResult) -> str:
        name = f"diagnosis-{time.strftime('%Y%m%d-%H%M%S')}"
        path = config.REPORTS_DIR / f"{name}.json"
        path.write_text(result.model_dump_json(indent=1))
        return name

    # ---- static ----------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ---- status ----------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        claude = ClaudeBackend()
        ollama = OllamaBackend()
        return {
            "default_backend": config.BACKEND,
            "backends": {
                "claude": {"model": config.MODEL, "reachable": claude.is_available(timeout=1.5), "host": config.API_BASE_URL},
                "ollama": {"model": config.LOCAL_MODEL, "reachable": ollama.is_available(timeout=1.0), "host": config.OLLAMA_HOST},
            },
            "knowledge_chunks": _store().count_chunks(),
            "queue_pending": sum(1 for j in q.list_jobs() if not j.done),
            "reports_dir": str(config.REPORTS_DIR),
        }

    # ---- vehicle ---------------------------------------------------------

    @app.get("/api/fixtures")
    def fixtures() -> list[dict[str, Any]]:
        out = []
        for f in sorted(config.FIXTURES_DIR.glob("*.json")):
            d = json.loads(f.read_text())
            v = d.get("vehicle", {})
            out.append(
                {
                    "name": f.name,
                    "label": " ".join(str(x) for x in (v.get("year"), v.get("make"), v.get("model")) if x)
                    + " · " + ", ".join(c["code"] for c in d.get("dtcs", [])),
                }
            )
        return out

    @app.post("/api/scan")
    def scan(body: dict[str, Any]) -> dict[str, Any]:
        fixture = body.get("fixture")
        port = body.get("port")
        if fixture:
            path = config.FIXTURES_DIR / Path(fixture).name
            if not path.exists():
                raise HTTPException(404, f"Fixture {fixture!r} not found")
            reader = open_reader(fixture=path)
        else:
            try:
                reader = open_reader(port=port or None, fast=not body.get("slow", False))
            except ConnectionError as e:
                raise HTTPException(503, str(e))
        try:
            return reader.read().model_dump()
        finally:
            reader.close()

    # ---- diagnose --------------------------------------------------------

    @app.post("/api/diagnose")
    async def diagnose(
        snapshot: str = Form(...),
        symptoms: str = Form(""),
        backend: str = Form(""),
        local_model: str = Form(""),
        offline: bool = Form(False),
        no_vin: bool = Form(False),
        strict: bool = Form(False),
        audio: UploadFile | None = File(None),
        images: list[UploadFile] = File([]),
    ):
        snap, p = await _prepare_from_form(snapshot, symptoms, no_vin, audio, images)
        base = {"flags": [f.model_dump() for f in p.flags], "chunks": [c.model_dump() for c in p.chunks]}

        if offline:
            return {"kind": "offline", "report": render_offline_report(snap, p.flags, p.chunks), **base}

        if local_model:
            config.LOCAL_MODEL = local_model
        chosen, note = choose_backend(backend or None)
        if not chosen.is_available():
            return {
                "kind": "offline",
                "note": f"{chosen.name} backend unreachable; showing the rules-only report.",
                "report": render_offline_report(snap, p.flags, p.chunks),
                **base,
            }
        try:
            result = call_model(p, backend=chosen, strict=strict)
        except DiagnosisRefused as e:
            raise HTTPException(422, str(e))
        except Offline as e:
            return {"kind": "offline", "note": str(e), "report": render_offline_report(snap, p.flags, p.chunks), **base}
        except (json.JSONDecodeError, pydantic.ValidationError) as e:
            raise HTTPException(502, f"Model output did not match the diagnosis schema: {str(e)[:300]}")
        except anthropic.AuthenticationError:
            raise HTTPException(401, "API credentials were rejected.")
        except TypeError as e:
            if "authentication method" in str(e):
                raise HTTPException(401, "No API credentials. Set ANTHROPIC_API_KEY or run `ant auth login`.")
            raise
        except anthropic.RateLimitError:
            raise HTTPException(429, "Rate limited; try again shortly or queue the job.")
        except anthropic.APIStatusError as e:
            raise HTTPException(502, f"API error {e.status_code}: {e.message}")
        except RuntimeError as e:
            raise HTTPException(502, str(e))

        name = _save_report(result)
        return {"kind": "diagnosis", "report_name": name, "note": note, "result": result.model_dump()}

    # ---- queue -----------------------------------------------------------

    @app.post("/api/queue")
    async def enqueue(
        snapshot: str = Form(...),
        symptoms: str = Form(""),
        no_vin: bool = Form(False),
        audio: UploadFile | None = File(None),
        images: list[UploadFile] = File([]),
    ):
        _, p = await _prepare_from_form(snapshot, symptoms, no_vin, audio, images)
        job = q.enqueue(p)
        return {"job": job.name, "path": str(job.path)}

    @app.get("/api/queue")
    def queue_list() -> list[dict[str, Any]]:
        out = []
        for j in q.list_jobs():
            err = j.path / "error.txt"
            out.append(
                {
                    "name": j.name,
                    "status": "done" if j.done else ("error" if err.exists() else "pending"),
                    "error": err.read_text().strip() if err.exists() else None,
                }
            )
        return out

    @app.post("/api/queue/drain")
    def drain(body: dict[str, Any] | None = None):
        chosen, note = choose_backend((body or {}).get("backend") or None)
        if not chosen.is_available():
            raise HTTPException(503, f"{chosen.name} backend unreachable; nothing sent.")
        try:
            outcomes = q.drain(backend=chosen)
        except TypeError as e:
            if "authentication method" in str(e):
                raise HTTPException(401, "No API credentials.")
            raise
        return {
            "note": note,
            "results": [
                {"job": job.name, "ok": r is not None, "error": err,
                 "top_cause": r.diagnosis.causes[0].title if r and r.diagnosis.causes else None}
                for job, r, err in outcomes
            ],
        }

    # ---- knowledge -------------------------------------------------------

    @app.get("/api/knowledge")
    def knowledge_list() -> list[dict[str, Any]]:
        return [dict(r) for r in _store().list_documents()]

    @app.post("/api/knowledge")
    async def knowledge_upload(files: list[UploadFile] = File(...)):
        results = []
        for f in files:
            if not f.filename:
                continue
            dst = config.HOME / "uploads" / Path(f.filename).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            for name, n, status in ingest_path(_store(), dst):
                results.append({"file": name, "chunks": n, "status": status})
        return {"results": results, "total_chunks": _store().count_chunks()}

    @app.post("/api/knowledge/seed")
    def knowledge_seed():
        results = ingest_path(_store(), config.SEED_KNOWLEDGE_DIR)
        return {"results": [{"file": n, "chunks": c, "status": s} for n, c, s in results], "total_chunks": _store().count_chunks()}

    @app.get("/api/search")
    def search(q_: str = "", k: int = 8):
        return [c.model_dump() for c in Retriever(_store(), top_k=k).search_text(q_, limit=k)]

    # ---- reports ---------------------------------------------------------

    @app.get("/api/reports")
    def reports() -> list[dict[str, Any]]:
        out = []
        for f in sorted(config.REPORTS_DIR.glob("diagnosis-*.json"), reverse=True)[:50]:
            try:
                d = json.loads(f.read_text())
                top = d["diagnosis"]["causes"][0]["title"] if d["diagnosis"]["causes"] else ""
                v = d["diagnosis"]
                out.append({"name": f.stem, "severity": v["severity"], "top_cause": top, "model": d.get("model", ""),
                            "backend": d.get("backend", "claude"), "mtime": f.stat().st_mtime})
            except (KeyError, json.JSONDecodeError):
                continue
        return out

    @app.get("/api/reports/{name}")
    def report(name: str):
        path = config.REPORTS_DIR / f"{Path(name).stem}.json"
        if not path.exists():
            raise HTTPException(404, "Report not found")
        return JSONResponse(json.loads(path.read_text()))

    @app.exception_handler(Exception)
    async def _unhandled(request, exc):  # noqa: ANN001
        return PlainTextResponse(f"Internal error: {type(exc).__name__}: {exc}", status_code=500)

    return app


app = create_app()
