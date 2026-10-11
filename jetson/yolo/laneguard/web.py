"""Operator web interface. Serves live view, polygon editor, events and stats from the running process.

Started by app.py with --web PORT. Also runnable standalone against a Runtime for tests.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from . import lcs
from .runtime import Runtime

log = logging.getLogger("laneguard.web")
STATIC = Path(__file__).parent / "static"


def build_app(rt: Runtime) -> FastAPI:
    app = FastAPI(title="Lane Guard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/state")
    def state():
        return JSONResponse(rt.snapshot())

    @app.get("/api/lane")
    def get_lane():
        return JSONResponse(rt.read_lane())

    @app.post("/api/lane")
    async def set_lane(req: Request):
        lane = await req.json()
        poly = lane.get("polygon")
        cd = lane.get("crew_direction")
        if not isinstance(poly, list) or len(poly) < 3 or not isinstance(cd, list) or len(cd) != 2:
            raise HTTPException(400, "polygon needs 3+ points and crew_direction needs 2 numbers")
        current = rt.read_lane()
        merged = {**current, **lane, "frame_size": list(rt.frame_size), "source": rt.source,
                  "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        rt.write_lane(merged)
        log.info("lane updated via web: %d points", len(poly))
        return {"ok": True, "lane": merged}

    @app.get("/api/lcs/districts")
    def lcs_districts():
        return {str(k): v for k, v in lcs.DISTRICTS.items()}

    @app.get("/api/lcs/closures")
    def lcs_closures(district: int, q: str = "", status: str = "", lat: float | None = None, lon: float | None = None,
                     radius_mi: float = 5.0, lane_only: bool = False, limit: int = 100, refresh: bool = False):
        try:
            recs = lcs.fetch_district(district, force=refresh)
        except Exception as e:  # offline box, DNS failure, Caltrans outage
            raise HTTPException(502, f"LCS fetch failed: {e}")
        return {"district": district, "total": len(recs),
                "results": lcs.search(recs, q=q, status=status, lat=lat, lon=lon, radius_mi=radius_mi,
                                      lane_only=lane_only, limit=limit)}

    @app.get("/api/site")
    def get_site():
        return JSONResponse(rt.read_site())

    @app.post("/api/site")
    async def set_site(req: Request):
        body = await req.json()
        site = rt.read_site()
        if "lcs" in body:
            site["lcs"] = body["lcs"] or {}
        for k in ("crew", "operator", "notes", "camera_lat", "camera_lon"):
            if k in body:
                site[k] = body[k]
        site = rt.write_site(site)
        log.info("site updated via web: closure %s", (site.get("lcs") or {}).get("index", "none"))
        return {"ok": True, "site": site}

    @app.post("/api/test_alarm")
    def test_alarm():
        rt.request_test_alarm()
        return {"ok": True}

    @app.get("/snapshot.jpg")
    def snapshot():
        with rt.lock:
            data = rt.raw_jpeg
        if not data:
            raise HTTPException(503, "no frame yet")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/stream.mjpg")
    def stream():
        def gen():
            last = None
            while True:
                with rt.lock:
                    data = rt.frame_jpeg
                if data and data is not last:
                    last = data
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                time.sleep(1 / 20)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/events/{name}")
    def event_file(name: str):
        p = (rt.evidence_dir / name).resolve()
        if not p.is_file() or rt.evidence_dir.resolve() not in p.parents:
            raise HTTPException(404)
        return FileResponse(p)

    return app


def start_in_thread(rt: Runtime, host: str = "0.0.0.0", port: int = 8080) -> threading.Thread:
    import uvicorn
    app = build_app(rt)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True, name="laneguard-web")
    th.start()
    log.info("web interface on http://%s:%d", host, port)
    return th
