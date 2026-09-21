"""Coarse engine-noise features with numpy only.

Claude does not take audio, so we reduce a recording to a few numbers the
reasoning layer can relate to RPM: level, dominant tones, and the rate of
repeating impulses (ticks, knocks). WAV is read natively; anything else is
converted with ffmpeg if available.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from ..models import AudioFeatures


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        width = w.getsampwidth()
        channels = w.getnchannels()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width {width}")
    data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if width == 1:
        data = (data - 128.0) / 128.0
    else:
        data = data / float(2 ** (8 * width - 1))
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def _to_wav(path: Path) -> Path:
    if path.suffix.lower() == ".wav":
        return path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"{path.suffix} audio needs ffmpeg on PATH (or convert to WAV first).")
    out = Path(tempfile.mkdtemp()) / (path.stem + ".wav")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "22050", str(out)],
        check=True,
    )
    return out


def dominant_frequencies(x: np.ndarray, rate: int, n: int = 4, fmin: float = 20.0, fmax: float = 4000.0) -> list[float]:
    if len(x) < 1024:
        return []
    window = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * window))
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
    mask = (freqs >= fmin) & (freqs <= fmax)
    spec, freqs = spec[mask], freqs[mask]
    if spec.size == 0:
        return []
    # Pick local maxima, strongest first, at least 15 Hz apart.
    peaks: list[float] = []
    order = np.argsort(spec)[::-1]
    for i in order:
        f = float(freqs[i])
        if 0 < i < len(spec) - 1 and spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1]:
            if all(abs(f - p) > 15 for p in peaks):
                peaks.append(round(f, 1))
        if len(peaks) >= n:
            break
    return peaks


def pulse_rate(x: np.ndarray, rate: int, min_hz: float = 2.0, max_hz: float = 80.0) -> float | None:
    """Rate of repeating impulses via autocorrelation of the amplitude envelope."""
    if len(x) < rate:  # need at least 1 s
        return None
    # Envelope: rectify, smooth to ~2 ms, downsample to 1 kHz.
    env = np.abs(x)
    k = max(1, int(rate * 0.002))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    step = max(1, rate // 1000)
    env = env[::step]
    env_rate = rate / step
    env = env - env.mean()
    if np.allclose(env, 0):
        return None
    ac = np.correlate(env, env, mode="full")[len(env) - 1 :]
    ac = ac / (ac[0] + 1e-12)
    lo = int(env_rate / max_hz)
    hi = int(env_rate / min_hz)
    if hi <= lo + 1 or hi >= len(ac):
        return None
    seg = ac[lo:hi]
    peak = float(np.max(seg))
    if peak < 0.25:  # weak periodicity: do not report
        return None
    # Autocorrelation also peaks at 2x, 3x the period. Take the shortest lag
    # that is a local maximum and within 80% of the strongest peak: that is
    # the fundamental period, not a subharmonic.
    for i in range(1, len(seg) - 1):
        if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] >= 0.8 * peak:
            return round(env_rate / (lo + i), 2)
    return round(env_rate / (lo + int(np.argmax(seg))), 2)


def analyze_audio(path: str | Path) -> AudioFeatures:
    src = Path(path)
    wav = _to_wav(src)
    x, rate = _load_wav(wav)
    duration = len(x) / rate
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    rms_db = 20 * np.log10(rms + 1e-9)
    notes: list[str] = []
    if duration < 3:
        notes.append("Recording is short (<3 s); features are unreliable.")
    if rms_db < -40:
        notes.append("Recording is very quiet; move the phone closer to the engine.")
    if np.max(np.abs(x)) > 0.98:
        notes.append("Recording clips; some features may be distorted.")
    return AudioFeatures(
        path=str(src),
        duration_s=round(duration, 2),
        rms_db=round(float(rms_db), 1),
        dominant_hz=dominant_frequencies(x, rate),
        pulse_rate_hz=pulse_rate(x, rate),
        notes=notes,
    )
