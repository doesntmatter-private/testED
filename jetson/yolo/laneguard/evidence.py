"""Rolling frame buffer and clip writer for event evidence."""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path

import cv2

log = logging.getLogger("laneguard.evidence")


class Evidence:
    def __init__(self, out_dir: Path, fps: float, pre_s: float = 5.0, post_s: float = 5.0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.pre = deque(maxlen=int(pre_s * fps))
        self.post_n = int(post_s * fps)
        self._active: list[tuple[Path, list, int, dict]] = []  # (path, frames, remaining, meta)

    def push(self, frame) -> None:
        self.pre.append(frame)
        still = []
        for path, frames, remaining, meta in self._active:
            frames.append(frame)
            remaining -= 1
            if remaining <= 0:
                self._write(path, frames, meta)
            else:
                still.append((path, frames, remaining, meta))
        self._active = still

    def start_clip(self, event, extra: dict | None = None) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(event.t)) if event.t > 1e9 else f"t{event.t:.2f}"
        path = self.out_dir / f"{stamp}_track{event.track_id}.mp4"
        meta = {"kind": event.kind, "track_id": event.track_id, "t": event.t,
                "anchor": event.anchor, "frames_inside": event.frames_inside,
                "speed_px_per_frame": event.speed_px_per_frame, **(extra or {})}
        self._active.append((path, list(self.pre), self.post_n, meta))
        return path

    def flush(self) -> None:
        """Write any clips still collecting post-event frames. Call at end of stream."""
        for path, frames, remaining, meta in self._active:
            meta = {**meta, "truncated_post_frames": remaining}
            self._write(path, frames, meta)
        self._active = []

    def _write(self, path: Path, frames: list, meta: dict) -> None:
        if not frames:
            return
        h, w = frames[0].shape[:2]
        # CPU encode. T400 has no NVENC; Jetson could use a GStreamer writer instead.
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        log.info("wrote evidence %s (%d frames)", path.name, len(frames))
