"""
ui/textures.py
OpenCV ↔ DearPyGui texture conversion utilities.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def bgr_to_dpg_flat(img: np.ndarray) -> list[float]:
    """Convert a BGR numpy image to a flat RGBA float32 list for DPG textures."""
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return (rgba.flatten().astype(np.float32) / 255.0).tolist()


def load_thumbnail(path: Optional[Path], size: int, seal_name: str, idx: int) -> np.ndarray:
    """
    Load a seal thumbnail from *path*, resizing to *size* × *size*.
    Falls back to a colour-coded placeholder if path is None or unreadable.
    """
    if path is not None:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            # If the image has an alpha channel, composite onto dark background
            if raw.shape[2] == 4:
                bgr   = raw[:, :, :3]
                alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
                bg    = np.full_like(bgr, 25)
                raw   = (bgr * alpha + bg * (1 - alpha)).astype(np.uint8)
            return cv2.resize(raw, (size, size))
        print(f"[WARN] Could not read thumbnail: {path}")

    return _make_placeholder_thumbnail(seal_name, idx, size)


def _make_placeholder_thumbnail(seal_name: str, idx: int, size: int) -> np.ndarray:
    """Generate a colour-coded placeholder thumbnail."""
    from core.config import Config  # avoid circular at module level
    n_seals = 12  # approximate; placeholder only
    hue     = int((idx / n_seals) * 179)
    colour  = cv2.cvtColor(np.uint8([[[hue, 160, 110]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()

    img = np.full((size, size, 3), [max(0, c // 4) for c in colour], dtype=np.uint8)
    cv2.rectangle(img, (1, 1), (size - 2, size - 2), colour, 2)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    (tw, th), _ = cv2.getTextSize(seal_name, font, scale, thick)
    cv2.putText(img, seal_name,
                ((size - tw) // 2, (size + th) // 2),
                font, scale, (220, 220, 220), thick)
    return img