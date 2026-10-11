"""Shared state between the processing loop and the web server."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path

import cv2

from . import lcs


class Runtime:
    def __init__(self, lane_path: Path, evidence_dir: Path, source: str, model: str):
        self.lock = threading.Lock()
        self.lane_path = Path(lane_path)
        self.evidence_dir = Path(evidence_dir)
        self.source = source
        self.model = model
        self.started = time.time()
        self.frame_jpeg: bytes | None = None       # annotated, for the live view
        self.raw_jpeg: bytes | None = None         # unannotated, for polygon drawing
        self.frame_size = (0, 0)
        self.frame_no = 0
        self.fps = 0.0
        self.infer_ms = 0.0
        self.tracks = 0
        self.last_alarm_t: float | None = None
        self.events: deque = deque(maxlen=200)
        self.lane_reload_requested = False
        self.test_alarm_requested = False
        self._jpeg_quality = 70

    # ---- called from the processing loop ----------------------------------

    def publish_frame(self, annotated, raw=None) -> None:
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        raw_buf = None
        if raw is not None:
            ok2, raw_buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self.lock:
            if ok:
                self.frame_jpeg = buf.tobytes()
            if raw_buf is not None:
                self.raw_jpeg = raw_buf.tobytes()
            self.frame_size = (annotated.shape[1], annotated.shape[0])

    def publish_stats(self, frame_no: int, fps: float, infer_ms: float, tracks: int) -> None:
        with self.lock:
            self.frame_no, self.fps, self.infer_ms, self.tracks = frame_no, fps, infer_ms, tracks

    def publish_event(self, event, clip_path: Path | None = None) -> None:
        d = asdict(event) if is_dataclass(event) else dict(event)
        d["clip"] = clip_path.name if clip_path else None
        d["wall_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d["t"])) if d["t"] > 1e9 else f"{d['t']:.2f}s"
        with self.lock:
            self.events.appendleft(d)
            if d["kind"] == "alarm":
                self.last_alarm_t = d["t"]

    def take_flags(self) -> tuple[bool, bool]:
        with self.lock:
            r, t = self.lane_reload_requested, self.test_alarm_requested
            self.lane_reload_requested = self.test_alarm_requested = False
            return r, t

    # ---- called from the web server --------------------------------------

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "source": self.source, "model": self.model,
                "uptime_s": round(time.time() - self.started, 1),
                "frame_no": self.frame_no, "fps": round(self.fps, 1), "infer_ms": round(self.infer_ms, 1),
                "tracks": self.tracks, "frame_size": self.frame_size,
                "last_alarm_t": self.last_alarm_t,
                "alarm_active": self.last_alarm_t is not None and (time.time() - self.last_alarm_t) < 3.0,
                "events": list(self.events)[:50],
                "lane": self.read_lane(),
                "site": self.read_site(),
            }

    def read_site(self) -> dict:
        return lcs.load_site(self.lane_path)

    def write_site(self, site: dict) -> dict:
        return lcs.save_site(self.lane_path, site)

    def read_lane(self) -> dict:
        try:
            return json.loads(self.lane_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def write_lane(self, lane: dict) -> None:
        self.lane_path.parent.mkdir(parents=True, exist_ok=True)
        self.lane_path.write_text(json.dumps(lane, indent=2))
        with self.lock:
            self.lane_reload_requested = True
            if self.raw_jpeg:
                self.lane_path.with_suffix(".snapshot.jpg").write_bytes(self.raw_jpeg)

    def request_test_alarm(self) -> None:
        with self.lock:
            self.test_alarm_requested = True
