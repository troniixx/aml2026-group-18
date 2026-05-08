"""
core/sequencer.py
Tracks the player's seal history and fires jutsu events when a combo is matched.
Supports an optional pre-effect announcement phase before the visual effect starts.
"""
from __future__ import annotations
from collections import deque
from typing import Optional


class JutsuSequencer:
    """
    Usage
    -----
    seq = JutsuSequencer(combos,
                         cooldown_frames=120,
                         effect_frames=90,
                         pre_effect_frames=60)   # 2 s announcement before effect

    # Once per frame:
    triggered = seq.update(detected_seal_label)   # None or jutsu name

    # Query current state:
    seq.active_jutsu      # str | None  — set during BOTH announce and effect phases
    seq.is_announcing     # bool        — True during the pre-effect window
    seq.announce_progress # float 0→1  — how far through the announcement we are
    seq.effect_progress   # float 0→1  — how far through the effect we are
    seq.effect_frame      # int         — absolute frame within effect (0 during announce)
    """

    def __init__(
        self,
        combos:            dict[str, tuple],
        cooldown_frames:   int = 120,
        effect_frames:     int = 90,
        pre_effect_frames: int = 0,
    ) -> None:
        self.combos            = combos
        self.cooldown_frames   = cooldown_frames
        self.effect_frames     = effect_frames
        self.pre_effect_frames = pre_effect_frames

        max_len = max(len(c) for c in combos.values())
        self._history: deque[str] = deque(maxlen=max_len)

        self._last_seal:           Optional[str] = None
        self._cooldown:            int           = 0
        self._active_jutsu:        Optional[str] = None
        self._effect_frame:        int           = 0
        self._pre_effect_countdown: int          = 0   # counts down from pre_effect_frames

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, seal_label: Optional[str]) -> Optional[str]:
        """
        Call once per render frame.
        Returns the jutsu name the moment it is triggered, else None.
        """
        if self._cooldown > 0:
            self._cooldown -= 1

        if self._active_jutsu is not None:
            if self._pre_effect_countdown > 0:
                # Still in announcement phase — tick down, don't advance effect
                self._pre_effect_countdown -= 1
            else:
                # Effect phase — advance frame counter
                self._effect_frame += 1
                if self._effect_frame >= self.effect_frames:
                    self._active_jutsu = None
                    self._effect_frame = 0

        # Append to history only on seal change
        if seal_label and seal_label != self._last_seal:
            self._history.append(seal_label)
            self._last_seal = seal_label
        elif not seal_label:
            self._last_seal = None

        # Check combos only when idle
        if self._cooldown == 0 and self._active_jutsu is None:
            match = self._check_combos()
            if match:
                self._active_jutsu        = match
                self._effect_frame        = 0
                self._pre_effect_countdown = self.pre_effect_frames
                self._cooldown            = self.cooldown_frames
                self._history.clear()
                self._last_seal           = None
                return match

        return None

    def reset(self) -> None:
        self._history.clear()
        self._last_seal           = None
        self._cooldown            = 0
        self._active_jutsu        = None
        self._effect_frame        = 0
        self._pre_effect_countdown = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def active_jutsu(self) -> Optional[str]:
        return self._active_jutsu

    @property
    def is_announcing(self) -> bool:
        """True during the pre-effect window (name shown, effect not yet playing)."""
        return self._active_jutsu is not None and self._pre_effect_countdown > 0

    @property
    def announce_progress(self) -> float:
        """0.0 at start of announcement, 1.0 when announcement ends."""
        if not self.is_announcing or self.pre_effect_frames == 0:
            return 0.0
        return 1 - self._pre_effect_countdown / self.pre_effect_frames

    @property
    def effect_progress(self) -> float:
        """0.0 at start of effect, 1.0 at end. 0.0 during announcement."""
        if self._active_jutsu is None or self.is_announcing:
            return 0.0
        return self._effect_frame / max(1, self.effect_frames)

    @property
    def effect_frame(self) -> int:
        """Absolute frame index within the effect. 0 during announcement."""
        return self._effect_frame if not self.is_announcing else 0

    @property
    def history(self) -> list[str]:
        return list(self._history)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_combos(self) -> Optional[str]:
        history = list(self._history)
        for name, combo in self.combos.items():
            n = len(combo)
            if len(history) >= n and tuple(history[-n:]) == tuple(combo):
                return name
        return None