"""Optional low light enhancement applied before inference. Cheap CPU ops, meant for night tests."""
from __future__ import annotations

import cv2
import numpy as np

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_gamma_lut = None
_gamma_val = None


def enhance(frame, gamma: float = 1.8, clahe: bool = True):
    """Gamma lift on dark frames plus CLAHE on luminance. Returns a new BGR frame."""
    global _gamma_lut, _gamma_val
    if _gamma_lut is None or _gamma_val != gamma:
        inv = 1.0 / gamma
        _gamma_lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
        _gamma_val = gamma
    out = cv2.LUT(frame, _gamma_lut)
    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out
