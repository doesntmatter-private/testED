import wave

import numpy as np

from autodiag.media.audio import analyze_audio, dominant_frequencies, pulse_rate


def _write_wav(path, x, rate=22050):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def test_dominant_frequency_detects_tone():
    rate = 22050
    t = np.arange(rate * 2) / rate
    x = 0.5 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    peaks = dominant_frequencies(x, rate)
    assert any(abs(p - 120) < 3 for p in peaks)
    assert any(abs(p - 440) < 3 for p in peaks)


def test_pulse_rate_detects_ticks():
    rate = 22050
    dur = 4
    x = np.random.default_rng(0).normal(0, 0.01, rate * dur)
    tick_hz = 12.0  # e.g. valvetrain tick at ~1440 rpm on a 4-cyl (rpm/60/2 * 2)
    for k in range(int(dur * tick_hz)):
        i = int(k * rate / tick_hz)
        x[i : i + 200] += 0.8 * np.exp(-np.arange(200) / 40.0)
    rate_hz = pulse_rate(x, rate)
    assert rate_hz is not None
    assert abs(rate_hz - tick_hz) < 0.6


def test_pulse_rate_none_for_noise():
    rate = 22050
    x = np.random.default_rng(1).normal(0, 0.1, rate * 3)
    assert pulse_rate(x, rate) is None


def test_analyze_audio_end_to_end(tmp_path):
    rate = 22050
    t = np.arange(rate * 3) / rate
    x = 0.3 * np.sin(2 * np.pi * 100 * t)
    p = tmp_path / "engine.wav"
    _write_wav(p, x, rate)
    f = analyze_audio(p)
    assert abs(f.duration_s - 3.0) < 0.01
    assert -20 < f.rms_db < -5
    assert any(abs(h - 100) < 3 for h in f.dominant_hz)
