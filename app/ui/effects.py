"""
ui/effects.py
Pure-OpenCV jutsu visual effects.

Each effect function signature:
    fn(frame, progress, frame_idx, hand_bbox, state) -> frame

    frame      : BGR ndarray, modified IN-PLACE and returned
    progress   : float 0.0 → 1.0  (where in the effect timeline we are)
    frame_idx  : int   0 → N-1    (absolute frame counter within the effect)
    hand_bbox  : Optional[tuple[int,int,int,int]]  (x1,y1,x2,y2) or None
    state      : dict  persisted across frames by the caller; reset on new jutsu

Public entry point:
    apply_effect(frame, jutsu_name, progress, frame_idx, hand_bbox, state)
"""
from __future__ import annotations
from typing import Optional

import cv2
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hand_center(frame: np.ndarray,
                 hand_bbox: Optional[tuple]) -> tuple[int, int]:
    fh, fw = frame.shape[:2]
    if hand_bbox is not None:
        x1, y1, x2, y2 = hand_bbox
        return (x1 + x2) // 2, (y1 + y2) // 2
    return fw // 2, fh // 2


def _ease_in_out(t: float) -> float:
    """Smooth 0→1 curve."""
    return t * t * (3 - 2 * t)


# ── 1. Fire Style: Fireball Jutsu ─────────────────────────────────────────────

def _fireball(frame, progress, frame_idx, hand_bbox, state):
    fh, fw = frame.shape[:2]
    cx, cy = _hand_center(frame, hand_bbox)

    # One-time particle initialisation with a fixed seed (deterministic)
    if 'particles' not in state:
        rng    = np.random.default_rng(7)
        n      = 120
        angles = rng.uniform(0, 2 * np.pi, n)
        speeds = rng.uniform(3, 14, n)
        sizes  = rng.integers(4, 20, n)
        # Fire palette in BGR: low-blue, high-green/red
        colors = [
            (int(rng.integers(0, 60)),
             int(rng.integers(80, 200)),
             int(rng.integers(180, 255)))
            for _ in range(n)
        ]
        state['particles'] = {
            'angles': angles, 'speeds': speeds,
            'sizes': sizes,   'colors': colors, 'n': n,
        }

    p      = state['particles']
    angles = p['angles']
    speeds = p['speeds']
    sizes  = p['sizes']
    colors = p['colors']
    n      = p['n']

    overlay = frame.copy()

    # ── Phase 1 (0-0.25): energy gather — glowing orb at hand ────────────────
    if progress < 0.25:
        t   = progress / 0.25
        orb = int(20 * _ease_in_out(t))
        cv2.circle(overlay, (cx, cy), orb + 12, (0, 60, 180), -1)
        cv2.circle(overlay, (cx, cy), orb + 6,  (0, 120, 240), -1)
        cv2.circle(overlay, (cx, cy), orb,       (100, 200, 255), -1)
        alpha = 0.6 * _ease_in_out(t)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    # ── Phase 2 (0.25-0.65): explosion ───────────────────────────────────────
    t_exp = (progress - 0.25) / 0.40
    # Screen flash at detonation
    if t_exp < 0.12:
        flash = np.full_like(frame, (60, 120, 255))
        cv2.addWeighted(flash, (0.12 - t_exp) / 0.12 * 0.55,
                        frame, 1.0, 0, frame)

    for i in range(n):
        travel = speeds[i] * frame_idx * 0.7
        px = int(cx + np.cos(angles[i]) * travel)
        py = int(cy + np.sin(angles[i]) * travel)
        if not (0 <= px < fw and 0 <= py < fh):
            continue
        life = max(0.0, 1.0 - t_exp)
        sz   = max(1, int(sizes[i] * life * 0.8))
        cv2.circle(overlay, (px, py), sz + 3, colors[i], -1)
        cv2.circle(overlay, (px, py), sz,     (200, 230, 255), -1)

    # Central fireball body
    body_r = int(fw * 0.12 * min(1.0, t_exp * 3))
    if body_r > 0:
        cv2.circle(overlay, (cx, cy), body_r,       (0,  80, 200), -1)
        cv2.circle(overlay, (cx, cy), body_r * 2//3,(0, 150, 255), -1)
        cv2.circle(overlay, (cx, cy), body_r // 3,  (150, 220, 255), -1)

    blend = min(0.9, t_exp * 2) * max(0.0, 1.0 - (t_exp - 0.5) * 2.5)
    cv2.addWeighted(overlay, blend, frame, 1 - blend, 0, frame)

    # Heat shimmer around explosion
    if 0.1 < t_exp < 0.8:
        r = int(fw * 0.18)
        x1b, y1b = max(0, cx - r), max(0, cy - r)
        x2b, y2b = min(fw, cx + r), min(fh, cy + r)
        roi = frame[y1b:y2b, x1b:x2b]
        if roi.size > 0:
            frame[y1b:y2b, x1b:x2b] = cv2.GaussianBlur(roi, (9, 9), 0)

    # ── Phase 3 (0.65-1.0): dissipation ──────────────────────────────────────
    if progress > 0.65:
        fade_alpha = (progress - 0.65) / 0.35 * 0.5
        dark = (frame * (1.0 - fade_alpha)).clip(0, 255).astype(np.uint8)
        frame[:] = dark

    return frame


# ── 2. Water Style: Hidden Mist Jutsu ─────────────────────────────────────────

def _hidden_mist(frame, progress, frame_idx, hand_bbox, state):
    fh, fw = frame.shape[:2]

    if 'mist' not in state:
        rng  = np.random.default_rng(3)
        base = rng.integers(210, 255, (fh, fw, 3), dtype=np.uint8)
        state['mist'] = cv2.GaussianBlur(base.astype(np.uint8), (101, 101), 0)

    mist = state['mist']

    # Alpha curve: ramp in 0→0.35, hold 0.35→0.75, ramp out 0.75→1.0
    if progress < 0.35:
        alpha = _ease_in_out(progress / 0.35) * 0.80
    elif progress < 0.75:
        alpha = 0.80
    else:
        alpha = 0.80 * (1.0 - _ease_in_out((progress - 0.75) / 0.25))

    # Desaturate frame under mist
    if alpha > 0.1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(gray, 0.35, frame, 0.65, 0, frame)

    cv2.addWeighted(mist, alpha, frame, 1.0 - alpha, 0, frame)

    # Slow horizontal mist streaks
    if 0.1 < alpha:
        for i in range(8):
            y      = int(fh * (i / 8.0) + np.sin(progress * 4 + i * 0.9) * 18)
            y      = max(0, min(fh - 1, y))
            x_end  = int(fw * min(1.0, progress * 2.5))
            thick  = int(np.random.uniform(1, 3))
            shade  = int(200 + alpha * 40)
            cv2.line(frame, (0, y), (x_end, y), (shade, shade, shade + 10), thick)

    return frame


# ── 3. Lightning Style: Chidori ───────────────────────────────────────────────

def _zigzag(p1: tuple, p2: tuple, roughness: float = 22, depth: int = 4) -> list:
    """Recursively subdivide a line into a jagged lightning path."""
    if depth == 0:
        return [p1, p2]
    mx = (p1[0] + p2[0]) // 2 + int(np.random.randn() * roughness)
    my = (p1[1] + p2[1]) // 2 + int(np.random.randn() * roughness)
    mid = (mx, my)
    return (
        _zigzag(p1, mid, roughness * 0.6, depth - 1) +
        _zigzag(mid, p2, roughness * 0.6, depth - 1)[1:]
    )


def _chidori(frame, progress, frame_idx, hand_bbox, state):
    fh, fw = frame.shape[:2]
    cx, cy = _hand_center(frame, hand_bbox)

    # Intensity envelope: ramp 0→0.4, peak 0.4→0.7, decay 0.7→1.0
    if progress < 0.4:
        intensity = _ease_in_out(progress / 0.4)
    elif progress < 0.7:
        intensity = 1.0
    else:
        intensity = 1.0 - _ease_in_out((progress - 0.7) / 0.3)

    glow = np.zeros_like(frame, dtype=np.float32)
    radius   = int(min(fw, fh) * 0.30 * intensity)
    n_bolts  = max(2, int(14 * intensity))

    for _ in range(n_bolts):
        angle = np.random.uniform(0, 2 * np.pi)
        ex    = cx + int(np.cos(angle) * radius)
        ey    = cy + int(np.sin(angle) * radius)
        pts   = _zigzag((cx, cy), (ex, ey), roughness=22, depth=4)
        pts   = [(np.clip(p[0], 0, fw-1), np.clip(p[1], 0, fh-1)) for p in pts]
        for i in range(len(pts) - 1):
            cv2.line(glow, pts[i], pts[i+1], (255, 200, 80), 2)   # blue-white bolt
            cv2.line(glow, pts[i], pts[i+1], (255, 255, 255), 1)  # bright core

    # Two-pass blur for soft halo
    glow = cv2.GaussianBlur(glow, (25, 25), 0)
    glow = cv2.GaussianBlur(glow, (11, 11), 0)

    # Additive blend onto frame
    result = np.clip(
        frame.astype(np.float32) + glow * intensity * 1.8,
        0, 255
    ).astype(np.uint8)

    # Blue tint during peak
    if intensity > 0.4:
        tint = np.zeros_like(result)
        tint[:, :, 0] = 80   # blue channel
        cv2.addWeighted(tint, (intensity - 0.4) * 0.35, result, 1.0, 0, result)

    # Orb at hand
    orb_r = int(32 * intensity)
    if orb_r > 1:
        cv2.circle(result, (cx, cy), orb_r + 10, (200, 130, 30), -1)
        cv2.circle(result, (cx, cy), orb_r + 4,  (255, 200, 80), -1)
        cv2.circle(result, (cx, cy), orb_r,       (255, 240, 160), -1)
        cv2.circle(result, (cx, cy), orb_r // 2,  (255, 255, 255), -1)

    frame[:] = result
    return frame


# ── 4. Wind Style: Great Breakthrough ─────────────────────────────────────────

def _wind_breakthrough(frame, progress, frame_idx, hand_bbox, state):
    fh, fw = frame.shape[:2]
    cx, cy = _hand_center(frame, hand_bbox)

    # Shockwave envelope: sharp rise 0→0.25, hold 0.25→0.55, decay 0.55→1.0
    if progress < 0.25:
        intensity = _ease_in_out(progress / 0.25)
    elif progress < 0.55:
        intensity = 1.0
    else:
        intensity = 1.0 - _ease_in_out((progress - 0.55) / 0.45)

    # ── Horizontal motion blur ────────────────────────────────────────────────
    k = max(3, int(51 * intensity))
    k = k if k % 2 == 1 else k + 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    blurred = cv2.filter2D(frame, -1, kernel)

    # Radial shockwave ring — blending strongest at the ring front
    Y, X       = np.ogrid[:fh, :fw]
    dist       = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
    max_dist   = np.sqrt(fw ** 2 + fh ** 2)
    ring_r     = max_dist * progress
    ring_width = max_dist * 0.28
    ring_mask  = np.exp(-((dist - ring_r) ** 2) / (2 * ring_width ** 2))
    ring_mask  = (ring_mask * intensity * 0.92).clip(0, 1)
    ring_3     = np.stack([ring_mask] * 3, axis=-1)

    frame = (blurred * ring_3 + frame * (1 - ring_3)).astype(np.uint8)

    # ── Speed streaks ─────────────────────────────────────────────────────────
    n_lines = int(25 * intensity)
    for _ in range(n_lines):
        y_s  = int(np.random.uniform(0, fh))
        lng  = int(np.random.uniform(fw * 0.25, fw * 0.85))
        x_s  = int(np.random.uniform(0, fw - lng))
        a    = np.random.uniform(0.25, 0.75) * intensity
        col  = (int(195 * a), int(215 * a), int(235 * a))
        cv2.line(frame, (x_s, y_s), (x_s + lng, y_s), col, 1)

    # ── Radial dust vignette ──────────────────────────────────────────────────
    if intensity > 0.3:
        Y2, X2 = np.ogrid[:fh, :fw]
        norm_d = np.sqrt(((X2 - fw/2)/(fw/2))**2 + ((Y2 - fh/2)/(fh/2))**2)
        dark   = 1.0 - np.clip((norm_d - 0.6), 0, 0.4) / 0.4 * intensity * 0.55
        frame  = (frame * np.stack([dark]*3, axis=-1)).clip(0, 255).astype(np.uint8)

    return frame


# ── 5. Shadow Clone Jutsu ─────────────────────────────────────────────────────

def _shadow_clone(frame, progress, frame_idx, hand_bbox, state):
    fh, fw = frame.shape[:2]

    # Ghost alpha: flash in 0→0.2, hold 0.2→0.75, fade 0.75→1.0
    if progress < 0.2:
        ghost_alpha = _ease_in_out(progress / 0.2) * 0.58
    elif progress < 0.75:
        ghost_alpha = 0.58
    else:
        ghost_alpha = 0.58 * (1.0 - _ease_in_out((progress - 0.75) / 0.25))

    # Clone offset: slides outward with progress, capped
    offset = int(fw * 0.20 * min(1.0, progress * 4))

    # Desaturated dark ghost
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ghost    = cv2.addWeighted(frame, 0.35, gray_bgr, 0.65, 0)
    # Blue-dark tint on ghost
    tint     = np.full_like(ghost, (20, 15, 5))
    cv2.addWeighted(tint, 0.2, ghost, 0.8, 0, ghost)

    result = frame.copy()
    for sign, ox in [(-1, -offset), (1, offset)]:
        if abs(ox) < 4:
            continue
        M       = np.float32([[1, 0, ox], [0, 1, 0]])
        shifted = cv2.warpAffine(ghost, M, (fw, fh))

        # Mask: only paint where the shifted clone lands
        mask = np.zeros((fh, fw), dtype=np.float32)
        if ox > 0:
            mask[:, ox:] = 1.0
        else:
            mask[:, :fw + ox] = 1.0

        mask_3 = np.stack([mask * ghost_alpha] * 3, axis=-1)
        result = (shifted * mask_3 + result * (1 - mask_3)).clip(0, 255).astype(np.uint8)

    # Bright flash at the moment of appearance
    if progress < 0.12:
        flash_a = (0.12 - progress) / 0.12 * 0.65
        white   = np.full_like(result, 255)
        cv2.addWeighted(white, flash_a, result, 1 - flash_a, 0, result)

    frame[:] = result
    return frame


# ── 6. Body Flicker Jutsu ─────────────────────────────────────────────────────

def _body_flicker(frame, progress, frame_idx, hand_bbox, state):
    fh, fw = frame.shape[:2]

    TRAIL_LEN = 10

    if 'trail' not in state:
        state['trail'] = []

    # Store a copy of the frame BEFORE effect is applied (clean trail)
    state['trail'].append(frame.copy())
    if len(state['trail']) > TRAIL_LEN:
        state['trail'].pop(0)

    # Phase 0.0-0.30: build-up — trail appears, person still visible
    # Phase 0.30-0.60: peak    — trail full, person mostly invisible
    # Phase 0.60-1.00: return  — trail fades, person reappears
    if progress < 0.30:
        trail_alpha = _ease_in_out(progress / 0.30)
        main_alpha  = 1.0
        n_trail     = max(1, int(len(state['trail']) * trail_alpha))
    elif progress < 0.60:
        trail_alpha = 1.0
        main_alpha  = 1.0 - _ease_in_out((progress - 0.30) / 0.30) * 0.88
        n_trail     = len(state['trail'])
    else:
        trail_alpha = 1.0 - _ease_in_out((progress - 0.60) / 0.40)
        main_alpha  = 0.12 + _ease_in_out((progress - 0.60) / 0.40) * 0.88
        n_trail     = len(state['trail'])

    # Composite: weighted sum of trail frames + current frame
    result = np.zeros((fh, fw, 3), dtype=np.float32)
    trail  = state['trail'][-n_trail:]
    for i, tf in enumerate(trail):
        w = (i + 1) / max(len(trail), 1) * trail_alpha * 0.55
        result += tf.astype(np.float32) * w
    result += frame.astype(np.float32) * main_alpha
    result  = result.clip(0, 255).astype(np.uint8)

    # Speed lines during vanish phase
    if 0.22 < progress < 0.68:
        cx, cy   = _hand_center(frame, hand_bbox)
        sp_int   = 1.0 - abs(progress - 0.45) / 0.23
        sp_int   = max(0.0, sp_int)
        for _ in range(int(18 * sp_int)):
            angle  = np.random.uniform(0, 2 * np.pi)
            r0, r1 = 25, 25 + int(np.random.uniform(40, 130))
            x1s    = cx + int(np.cos(angle) * r0)
            y1s    = cy + int(np.sin(angle) * r0)
            x2s    = cx + int(np.cos(angle) * r1)
            y2s    = cy + int(np.sin(angle) * r1)
            a      = np.random.uniform(0.35, 0.75) * sp_int
            col    = (int(190 * a), int(195 * a), int(200 * a))
            cv2.line(result, (x1s, y1s), (x2s, y2s), col, 1)

    frame[:] = result
    return frame


# ── Dispatch ──────────────────────────────────────────────────────────────────

_EFFECTS: dict = {
    "Fire Style: Fireball Jutsu":      _fireball,
    "Water Style: Hidden Mist Jutsu":  _hidden_mist,
    "Lightning Style: Chidori":        _chidori,
    "Wind Style: Great Breakthrough":  _wind_breakthrough,
    "Shadow Clone Jutsu":              _shadow_clone,
    "Body Flicker Jutsu":              _body_flicker,
}


def apply_effect(
    frame:      np.ndarray,
    jutsu_name: str,
    progress:   float,
    frame_idx:  int,
    hand_bbox:  Optional[tuple],
    state:      dict,
) -> np.ndarray:
    """
    Dispatch to the correct effect and return the modified frame.
    `state` is a plain dict the caller keeps alive for the effect's lifetime.
    """
    fn = _EFFECTS.get(jutsu_name)
    if fn is None:
        return frame
    return fn(frame, progress, frame_idx, hand_bbox, state)