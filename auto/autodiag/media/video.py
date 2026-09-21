"""Sample a handful of frames from a video with ffmpeg so they can go in as images."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def extract_frames(video: str | Path, n: int = 4, out_dir: str | Path | None = None) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        raise RuntimeError("Video input needs ffmpeg on PATH.")
    video = Path(video)
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="autodiag_frames_"))
    out.mkdir(parents=True, exist_ok=True)

    duration = None
    if ffprobe:
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
                capture_output=True, text=True, check=True,
            )
            duration = float(r.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            duration = None

    paths: list[str] = []
    if duration and duration > 0:
        for i in range(n):
            t = duration * (i + 0.5) / n
            dst = out / f"{video.stem}_frame{i + 1}.jpg"
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-q:v", "3", str(dst)],
                check=True,
            )
            paths.append(str(dst))
    else:
        # Unknown duration: take one frame per 2 seconds up to n.
        pattern = out / f"{video.stem}_frame%d.jpg"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-vf", "fps=1/2", "-frames:v", str(n), "-q:v", "3", str(pattern)],
            check=True,
        )
        paths = sorted(str(p) for p in out.glob(f"{video.stem}_frame*.jpg"))
    return paths
