"""
ui/app.py
DearPyGui interface for the Naruto Hand Seal Detector.
Receives DetectionResult objects from core.Detector and renders them.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from typing import Optional
from datetime import datetime
import os
import subprocess

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from core.config import Config
from core.detector import Detector, DetectionResult
from core.sequencer import JutsuSequencer
from ui.effects import apply_effect
from ui.textures import bgr_to_dpg_flat, load_thumbnail


EFFECT_COMBOS = {
    "Fire Style: Fireball Jutsu":      ("snake", "ram", "monkey", "boar", "horse", "tiger"),
    "Water Style: Hidden Mist Jutsu":  ("ox", "monkey", "rat", "ram", "snake", "dragon"),
    "Lightning Style: Chidori":        ("ox", "hare", "monkey"),
    "Wind Style: Great Breakthrough":  ("tiger", "ox", "dog", "hare", "bird"),
    "Shadow Clone Jutsu":              ("ram", "snake", "tiger"),
    "Body Flicker Jutsu":              ("ram", "horse", "snake"),
}

CHEATSHEET_PANEL_WIDTH = 260
INFO_PANEL_WIDTH       = 220
RECORDING_BAR_HEIGHT   = 38

SEAL_COOLDOWN_FRAMES   = 5
VIEWPORT_HEIGHT_PAD    = 120
PRE_EFFECT_FRAMES      = 10   # 60 = 2 s announcement before effect plays

RECORD_BTN_COLOR        = (30, 100, 45)
RECORD_BTN_COLOR_HOVER  = (40, 130, 60)
RECORD_BTN_COLOR_ACTIVE = (50, 150, 70)

BLINK_INTERVAL = 18

BORDER_PINK  = (150,  80, 200)
BORDER_GREEN = ( 50, 200,  80)


class App:
    def __init__(self, cfg: Config, detector: Detector) -> None:
        self.cfg      = cfg
        self.detector = detector
        self._seal_cooldown: int = 0

        self.cap: cv2.VideoCapture = cv2.VideoCapture(cfg.ui.camera_index)
        ret, probe = self.cap.read()
        if not ret:
            raise RuntimeError("Could not open camera.")
        self.frame_h, self.frame_w = probe.shape[:2]

        self._ghost_img:  Optional[np.ndarray] = None
        self._show_ghost: bool                  = False

        self._jutsu_sequence: Optional[tuple] = None
        self._jutsu_name:     Optional[str]   = None

        self._seq_match_idx:   int           = 0
        self._last_seq_label:  Optional[str] = None

        # Track announcement→effect transition to reset sequence highlights
        self._was_announcing: bool = False

        self._seal_cv2_thumbs: dict[str, np.ndarray] = {}

        self.JUTSU_COMBOS = {
            name: tuple(seals)
            for name, seals in cfg.jutsu_combos.items()
        }

        self.sequencer = JutsuSequencer(
            EFFECT_COMBOS,
            cooldown_frames=120,
            effect_frames=90,
            pre_effect_frames=PRE_EFFECT_FRAMES,   # 2 s announcement
        )
        self.effect_state: dict = {}

        self._last_result: Optional[DetectionResult] = None

        self._recording:      bool                      = False
        self._video_writer:   Optional[cv2.VideoWriter] = None
        self._recording_path: Optional[Path]            = None
        self._blink_frame:    int                       = 0
        self._blink_visible:  bool                      = True

        self.C_PRIMARY   = tuple(cfg.ui.primary_color)
        self.C_SECONDARY = tuple(cfg.ui.secondary_color)
        self.C_DIM       = (100, 100, 100)
        self.C_WHITE     = (220, 220, 220)
        self.C_BG_DARK   = ( 15,  15,  15)
        self.C_BG_MID    = ( 25,  25,  25)
        self.C_RED       = (200,  60,  60)

        self._seal_history: list[str] = []
        self._last_seal:    Optional[str] = None

        self._setup_dpg(probe)

    # ──────────────────────────────────────────────────────────────────────────
    #  Setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_dpg(self, probe: np.ndarray) -> None:
        dpg.create_context()
        self._build_theme()
        self._load_fonts()
        self._register_textures(probe)
        self._build_ui()

        dpg.create_viewport(
            title="Naruto Hand Seal Detection",
            width=self.frame_w + INFO_PANEL_WIDTH + CHEATSHEET_PANEL_WIDTH,
            height=self.frame_h + VIEWPORT_HEIGHT_PAD,
            resizable=True,
            min_width=640,
            min_height=360,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("win_main", True)

    def _load_fonts(self) -> None:
        self._font_large = None
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
        with dpg.font_registry():
            for path in candidates:
                if Path(path).exists():
                    self._font_large = dpg.add_font(path, 26)
                    break

    def _build_theme(self) -> None:
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       self.C_BG_DARK)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        self.C_BG_MID)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (35, 35, 35))
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram,  self.C_PRIMARY)
                dpg.add_theme_color(dpg.mvThemeCol_Text,           self.C_WHITE)
                dpg.add_theme_color(dpg.mvThemeCol_Button,         (40, 40, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (60, 60, 60))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (80, 80, 80))
                dpg.add_theme_color(dpg.mvThemeCol_Header,         (35, 35, 35))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,  (50, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,   (65, 65, 65))
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,  4)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,    8, 6)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,  8, 8)
        dpg.bind_theme(global_theme)

        with dpg.theme() as self._theme_record:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        RECORD_BTN_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, RECORD_BTN_COLOR_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  RECORD_BTN_COLOR_ACTIVE)

        with dpg.theme() as self._theme_red:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (130, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (160, 40, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (180, 50, 50))

        with dpg.theme() as self._theme_save:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (40,  60, 100))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (55,  80, 130))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (70, 100, 160))

    def _register_textures(self, probe: np.ndarray) -> None:
        size = self.cfg.ui.seal_thumb_size
        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                self.frame_w, self.frame_h,
                bgr_to_dpg_flat(probe),
                tag="tex_camera",
            )

            logo_path = Path(__file__).parent.parent.parent / "assets" / "Naruto_hand_seal_detector_title.png"
            logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
            if logo is None:
                raise FileNotFoundError(f"Logo not found at: {logo_path.resolve()}")
            if len(logo.shape) == 3 and logo.shape[2] == 4:
                alpha = logo[:, :, 3:4] / 255.0
                bg    = np.full_like(logo[:, :, :3], 25)
                logo  = (logo[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
            target_width   = INFO_PANEL_WIDTH - 16
            orig_h, orig_w = logo.shape[:2]
            target_height  = int(target_width * orig_h / orig_w)
            logo = cv2.resize(logo, (target_width, target_height))
            dpg.add_static_texture(logo.shape[1], logo.shape[0],
                                   bgr_to_dpg_flat(logo), tag="tex_logo")

            for idx, seal in enumerate(self.cfg.all_seals):
                thumb = load_thumbnail(self.cfg.seal_images.get(seal), size, seal, idx)
                self._seal_cv2_thumbs[seal] = thumb.copy()
                dpg.add_static_texture(size, size, bgr_to_dpg_flat(thumb),
                                       tag=f"tex_seal_{seal}")

    def _build_ui(self) -> None:
        fw, fh = self.frame_w, self.frame_h
        pw     = INFO_PANEL_WIDTH
        cw     = CHEATSHEET_PANEL_WIDTH

        with dpg.window(tag="win_main", no_title_bar=True, no_resize=True,
                        no_move=True, no_scrollbar=True,
                        width=fw + pw + cw, height=fh, pos=(0, 0)):

            with dpg.group(horizontal=True):

                with dpg.child_window(tag="child_left",
                                      width=fw, height=fh,
                                      no_scrollbar=True, border=False):
                    dpg.add_image("tex_camera",
                                  width=fw,
                                  height=fh - RECORDING_BAR_HEIGHT,
                                  tag="img_feed")
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="  ● REC  ", tag="btn_record",
                                       callback=self._start_recording)
                        dpg.bind_item_theme("btn_record", self._theme_record)
                        dpg.add_button(label="  stop  ", tag="btn_stop",
                                       callback=self._stop_recording)
                        dpg.bind_item_theme("btn_stop", self._theme_red)
                        dpg.hide_item("btn_stop")
                        dpg.add_button(label="  open recordings folder  ",
                                       tag="btn_save",
                                       callback=self._open_recordings_folder)
                        dpg.bind_item_theme("btn_save", self._theme_save)
                        dpg.hide_item("btn_save")

                with dpg.child_window(tag="child_right",
                                      width=pw, height=fh,
                                      no_scrollbar=False, border=False):
                    with dpg.group(horizontal=True):
                        dpg.add_image("tex_logo")
                    dpg.add_text("", tag="lbl_active_model", color=self.C_DIM)
                    dpg.add_separator()
                    dpg.add_spacer(height=2)

                    dpg.add_text("MODEL", color=self.C_PRIMARY)
                    model_names = [m.name for m in self.cfg.models.values()]
                    model_keys  = list(self.cfg.models.keys())
                    active_idx  = model_keys.index(self.cfg.active_model)
                    dpg.add_combo(items=model_names,
                                  default_value=model_names[active_idx],
                                  width=-1, tag="combo_model",
                                  callback=self._on_model_switch,
                                  user_data=model_keys)
                    dpg.add_spacer(height=2)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    dpg.add_text("CURRENT SEAL", color=self.C_PRIMARY)
                    dpg.add_spacer(height=2)
                    with dpg.group(horizontal=True):
                        dpg.add_text(" — ", tag="lbl_seal",     color=self.C_SECONDARY)
                        dpg.add_text("",    tag="lbl_conf_pct", color=self.C_DIM)
                    if self._font_large is not None:
                        dpg.bind_item_font("lbl_seal",     self._font_large)
                        dpg.bind_item_font("lbl_conf_pct", self._font_large)

                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    dpg.add_text("SEAL HISTORY", color=self.C_PRIMARY)
                    dpg.add_spacer(height=2)
                    for i in range(5):
                        dpg.add_text("", tag=f"lbl_hist_{i}")

                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    dpg.add_text("ACTIVE JUTSU", color=self.C_PRIMARY)
                    dpg.add_spacer(height=2)
                    dpg.add_text("—", tag="lbl_active_jutsu", color=self.C_SECONDARY)

                    dpg.add_spacer(height=8)
                    dpg.add_separator()
                    dpg.add_text("Press  Q  to quit", color=self.C_DIM)

                with dpg.child_window(tag="child_cheatsheet",
                                      width=cw, height=fh,
                                      no_scrollbar=False, border=False):
                    dpg.add_text("CHEATSHEETS", color=self.C_PRIMARY)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    with dpg.collapsing_header(label="Seal Cheatsheet",
                                               default_open=True):
                        dpg.add_spacer(height=2)
                        dpg.add_text("Click to overlay on feed", color=self.C_DIM)
                        dpg.add_spacer(height=4)
                        COLS   = 2
                        size   = self.cfg.ui.seal_thumb_size
                        chunks = [self.cfg.all_seals[i:i + COLS]
                                  for i in range(0, len(self.cfg.all_seals), COLS)]
                        with dpg.table(header_row=False,
                                       borders_innerH=False, borders_outerH=False,
                                       borders_innerV=False, borders_outerV=False,
                                       pad_outerX=False):
                            for _ in range(COLS):
                                dpg.add_table_column()
                            for chunk in chunks:
                                with dpg.table_row():
                                    for seal in chunk:
                                        with dpg.table_cell():
                                            dpg.add_image_button(
                                                f"tex_seal_{seal}",
                                                width=size, height=size,
                                                callback=lambda s, a, u: self._load_ghost(u),
                                                user_data=seal,
                                            )
                                            dpg.add_text(seal.capitalize(), color=self.C_DIM)
                        dpg.add_spacer(height=4)
                        dpg.add_button(label="Clear overlay", width=-1,
                                       callback=lambda: self._clear_ghost())

                    dpg.add_spacer(height=4)
                    dpg.add_separator()
                    dpg.add_spacer(height=4)

                    with dpg.collapsing_header(label="Jutsu Cheatsheet",
                                               default_open=True):
                        dpg.add_spacer(height=2)
                        dpg.add_text("Click to show seal sequence\non camera feed",
                                     color=self.C_DIM)
                        dpg.add_spacer(height=4)
                        for jutsu_name, combo in self.JUTSU_COMBOS.items():
                            dpg.add_button(label=jutsu_name, width=-1,
                                           callback=self._on_jutsu_selected,
                                           user_data=(jutsu_name, combo))
                            dpg.add_spacer(height=2)
                        dpg.add_spacer(height=4)
                        dpg.add_button(label="Clear sequence", width=-1,
                                       callback=lambda: self._clear_jutsu_sequence())

    # ──────────────────────────────────────────────────────────────────────────
    #  Recording
    # ──────────────────────────────────────────────────────────────────────────

    def _start_recording(self) -> None:
        recordings_dir = Path(__file__).parent.parent / "recordings"
        recordings_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._recording_path = recordings_dir / f"recording_{timestamp}.mp4"
        fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer   = cv2.VideoWriter(
            str(self._recording_path), fourcc, fps, (self.frame_w, self.frame_h))
        self._recording     = True
        self._blink_frame   = 0
        self._blink_visible = True
        dpg.hide_item("btn_record")
        dpg.show_item("btn_stop")
        dpg.hide_item("btn_save")

    def _stop_recording(self) -> None:
        self._recording = False
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        dpg.show_item("btn_record")
        dpg.hide_item("btn_stop")
        dpg.show_item("btn_save")

    def _open_recordings_folder(self) -> None:
        recordings_dir = Path(__file__).parent.parent / "recordings"
        if self._recording_path and self._recording_path.exists():
            subprocess.Popen(f'explorer /select,"{self._recording_path}"')
        elif recordings_dir.exists():
            os.startfile(str(recordings_dir))

    def _draw_rec_indicator(self, frame: np.ndarray) -> np.ndarray:
        if not self._recording:
            return frame
        self._blink_frame += 1
        if self._blink_frame >= BLINK_INTERVAL:
            self._blink_frame   = 0
            self._blink_visible = not self._blink_visible
        if not self._blink_visible:
            return frame
        cv2.circle(frame, (22, 22), 10, (0, 0, 220), -1)
        cv2.circle(frame, (22, 22), 10, (0, 0, 0), 1)
        cv2.putText(frame, "REC", (38, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA)
        return frame

    # ──────────────────────────────────────────────────────────────────────────
    #  Ghost overlay
    # ──────────────────────────────────────────────────────────────────────────

    def _load_ghost(self, seal_name: str) -> None:
        self._jutsu_sequence = None
        self._jutsu_name     = None
        path = (self.cfg.ghost_images.get(seal_name)
                or self.cfg.seal_images.get(seal_name))
        if path:
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            img = raw if raw is not None else self._make_pip_placeholder(seal_name)
        else:
            img = self._make_pip_placeholder(seal_name)
        self._ghost_img  = img
        self._show_ghost = True

    def _make_pip_placeholder(self, seal_name: str) -> np.ndarray:
        size = 200
        img  = np.full((size, size, 3), 50, dtype=np.uint8)
        cv2.rectangle(img, (2, 2), (size - 3, size - 3), (120, 120, 120), 2)
        font, scale = cv2.FONT_HERSHEY_DUPLEX, 0.55
        (tw, th), _ = cv2.getTextSize(seal_name.upper(), font, scale, 1)
        cv2.putText(img, seal_name.upper(),
                    ((size - tw) // 2, (size + th) // 2),
                    font, scale, (200, 200, 200), 1)
        return img

    def _clear_ghost(self) -> None:
        self._show_ghost = False

    def _apply_ghost(self, frame: np.ndarray) -> np.ndarray:
        if not self._show_ghost or self._ghost_img is None:
            return frame
        fh, fw   = frame.shape[:2]
        pip_size = max(80, min(200, int(min(fh, fw) * 0.30)))
        margin   = 12
        thumb = cv2.resize(self._ghost_img, (pip_size, pip_size))
        x1 = fw - pip_size - margin
        y1 = fh - pip_size - margin
        x2 = x1 + pip_size
        y2 = y1 + pip_size
        if len(thumb.shape) == 3 and thumb.shape[2] == 4:
            bgr   = thumb[:, :, :3]
            alpha = thumb[:, :, 3:4].astype(np.float32) / 255.0
            roi   = frame[y1:y2, x1:x2].astype(np.float32)
            frame[y1:y2, x1:x2] = (bgr * alpha + roi * (1 - alpha)).astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = thumb
        border_color = (self.C_PRIMARY[2], self.C_PRIMARY[1], self.C_PRIMARY[0])
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), border_color, 2)
        return frame

    # ──────────────────────────────────────────────────────────────────────────
    #  Jutsu sequence preview strip
    # ──────────────────────────────────────────────────────────────────────────

    def _on_jutsu_selected(self, sender, app_data, user_data) -> None:
        self._show_ghost     = False
        self._jutsu_name, self._jutsu_sequence = user_data
        self._seq_match_idx  = 0
        self._last_seq_label = None

    def _clear_jutsu_sequence(self) -> None:
        self._jutsu_sequence  = None
        self._jutsu_name      = None
        self._seq_match_idx   = 0
        self._last_seq_label  = None

    def _update_sequence_progress(self, label: Optional[str]) -> None:
        if not self._jutsu_sequence:
            return
        if label == self._last_seq_label:
            return
        self._last_seq_label = label
        if label is None:
            return

        if self._seq_match_idx < len(self._jutsu_sequence):
            expected = self._jutsu_sequence[self._seq_match_idx]
            if label == expected:
                self._seq_match_idx += 1
                # Do NOT reset when complete — stay at len(seq) so all borders
                # stay green during the announcement phase
            else:
                self._seq_match_idx = 0

    def _draw_jutsu_sequence(self, frame: np.ndarray) -> np.ndarray:
        if not self._jutsu_sequence:
            return frame

        fh, fw     = frame.shape[:2]
        n          = len(self._jutsu_sequence)
        thumb_size = min(64, (fw - 20) // max(n, 1))
        label_h    = 18
        bar_h      = thumb_size + label_h + 20
        bar_y      = fh - bar_h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y - 6), (fw, fh), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        prim_bgr = (self.C_PRIMARY[2], self.C_PRIMARY[1], self.C_PRIMARY[0])
        cv2.putText(frame, self._jutsu_name or "",
                    (10, bar_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    prim_bgr, 1, cv2.LINE_AA)

        total_w = n * thumb_size + (n - 1) * 6
        start_x = (fw - total_w) // 2

        for i, seal in enumerate(self._jutsu_sequence):
            x = start_x + i * (thumb_size + 6)
            y = bar_y + 14

            if seal in self._seal_cv2_thumbs:
                thumb = cv2.resize(self._seal_cv2_thumbs[seal], (thumb_size, thumb_size))
                frame[y:y + thumb_size, x:x + thumb_size] = thumb
            else:
                cv2.rectangle(frame, (x, y), (x + thumb_size, y + thumb_size),
                              (60, 60, 60), -1)

            border    = BORDER_GREEN if i < self._seq_match_idx else BORDER_PINK
            thickness = 2            if i < self._seq_match_idx else 1
            cv2.rectangle(frame,
                          (x - 2, y - 2),
                          (x + thumb_size + 2, y + thumb_size + 2),
                          border, thickness)

            if i < n - 1:
                ax = x + thumb_size + 2
                ay = y + thumb_size // 2
                cv2.arrowedLine(frame, (ax, ay), (ax + 4, ay),
                                (160, 160, 160), 1, tipLength=0.5)

            lbl = seal.capitalize()
            (lw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.putText(frame, lbl,
                        (x + (thumb_size - lw) // 2, y + thumb_size + label_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    #  Jutsu announcement overlay
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_jutsu_announcement(self, frame: np.ndarray) -> np.ndarray:
        """
        During the pre-effect window, draw the jutsu name centered at the top
        of the camera feed with a fade-in, hold, fade-out envelope.
        """
        if not self.sequencer.is_announcing:
            return frame

        fh, fw   = frame.shape[:2]
        progress = self.sequencer.announce_progress   # 0 → 1

        # Fade in over first 20%, hold, fade out over last 20%
        if progress < 0.20:
            alpha = progress / 0.20
        elif progress > 0.80:
            alpha = (1.0 - progress) / 0.20
        else:
            alpha = 1.0
        alpha = float(np.clip(alpha, 0.0, 1.0))

        jutsu_text = (self.sequencer.active_jutsu or "").upper()
        font       = cv2.FONT_HERSHEY_SIMPLEX
        scale      = 1.1
        thickness  = 2

        (tw, th), _ = cv2.getTextSize(jutsu_text, font, scale, thickness)
        tx = (fw - tw) // 2
        ty = 62

        # Semi-transparent dark banner
        # banner = frame.copy()
        # cv2.rectangle(banner, (0, 0), (fw, ty + 18), (5, 5, 5), -1)
        # cv2.addWeighted(banner, 0.65 * alpha, frame, 1 - 0.65 * alpha, 0, frame)

        # Jutsu name text blended onto frame
        text_layer = frame.copy()
        prim_bgr   = (self.C_PRIMARY[2], self.C_PRIMARY[1], self.C_PRIMARY[0])
        cv2.putText(text_layer, jutsu_text, (tx, ty),
                    font, scale, prim_bgr, thickness, cv2.LINE_AA)
        cv2.addWeighted(text_layer, alpha, frame, 1.0 - alpha, 0, frame)

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    #  UI updates
    # ──────────────────────────────────────────────────────────────────────────

    def _on_model_switch(self, sender, app_data, user_data) -> None:
        model_keys  = user_data
        model_names = [m.name for m in self.cfg.models.values()]
        key = model_keys[model_names.index(app_data)]
        self.detector.switch_model(key)
        self.sequencer.reset()
        self.effect_state.clear()
        dpg.set_value("lbl_active_model", f"Active: {self.cfg.models[key].name}")

    def _update_ui(self, result: DetectionResult) -> None:
        SEAL_HISTORY = 5
        label_text = result.label.upper() if result.label else "—"
        conf_text  = f"  {result.confidence:.0%}" if result.label else ""
        dpg.set_value("lbl_seal",     label_text)
        dpg.set_value("lbl_conf_pct", conf_text)

        if result.label and result.label != self._last_seal:
            self._seal_history.insert(0, result.label)
            self._seal_history = self._seal_history[:SEAL_HISTORY]
            self._last_seal    = result.label

        for i in range(SEAL_HISTORY):
            if i < len(self._seal_history):
                seal   = self._seal_history[i]
                prefix = "▶  " if i == 0 else "     "
                fade   = max(60, 220 - i * 40)
                color  = self.C_PRIMARY if i == 0 else (fade, fade, fade)
                dpg.set_value(f"lbl_hist_{i}",      f"{prefix}{seal.capitalize()}")
                dpg.configure_item(f"lbl_hist_{i}", color=color)
            else:
                dpg.set_value(f"lbl_hist_{i}", "")

        jutsu = self.sequencer.active_jutsu
        dpg.set_value("lbl_active_jutsu", jutsu.upper() if jutsu else "—")

    def _resize_to_viewport(self) -> None:
        vp_w = dpg.get_viewport_client_width()
        vp_h = dpg.get_viewport_client_height()
        left_w     = max(1, vp_w - INFO_PANEL_WIDTH - CHEATSHEET_PANEL_WIDTH - 2)
        left_h     = max(1, vp_h)
        img_area_h = max(1, left_h - RECORDING_BAR_HEIGHT)
        aspect = self.frame_w / self.frame_h
        if left_w / img_area_h > aspect:
            img_h = img_area_h
            img_w = int(img_h * aspect)
        else:
            img_w = left_w
            img_h = int(img_w / aspect)
        dpg.configure_item("child_left",       width=left_w,                 height=left_h)
        dpg.configure_item("child_right",      width=INFO_PANEL_WIDTH,       height=left_h)
        dpg.configure_item("child_cheatsheet", width=CHEATSHEET_PANEL_WIDTH, height=left_h)
        dpg.configure_item("img_feed",         width=img_w,                  height=img_h)

    def _draw_detection(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        # ── Fix 3: hide bbox while a jutsu effect is playing ─────────────────
        if self.sequencer.active_jutsu and not self.sequencer.is_announcing:
            return frame

        if result.hand_bbox is None:
            return frame

        x1, y1, x2, y2 = result.hand_bbox
        bgr = (self.C_PRIMARY[2], self.C_PRIMARY[1], self.C_PRIMARY[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        label = f"{result.label}  {result.confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 16), (x1 + lw + 10, y1), bgr, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return frame

    # ──────────────────────────────────────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            while dpg.is_dearpygui_running():
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # ── Model inference — always runs for live hand tracking ───────
                live_result = self.detector.predict(frame)

                if self.sequencer.active_jutsu and not self.sequencer.is_announcing:
                    # Effect playing — freeze label, bbox stays live
                    frozen = self._last_result or live_result
                    result = DetectionResult(
                        label      = frozen.label,
                        confidence = frozen.confidence,
                        hand_bbox  = live_result.hand_bbox,
                        history    = frozen.history,
                        jutsu      = frozen.jutsu,
                    )
                elif self._seal_cooldown > 0:
                    # Cooldown — freeze label, bbox stays live
                    self._seal_cooldown -= 1
                    frozen = self._last_result or live_result
                    result = DetectionResult(
                        label      = frozen.label,
                        confidence = frozen.confidence,
                        hand_bbox  = live_result.hand_bbox,
                        history    = frozen.history,
                        jutsu      = frozen.jutsu,
                    )
                else:
                    result            = live_result
                    self._last_result = result
                    if result.label is not None:
                        self._seal_cooldown = SEAL_COOLDOWN_FRAMES

                # ── Sequence progress ─────────────────────────────────────────
                # Update during normal detection and during announcement
                # (so the last seal lights up green before the effect starts)
                if not (self.sequencer.active_jutsu and not self.sequencer.is_announcing):
                    self._update_sequence_progress(result.label)

                # ── Sequencer ─────────────────────────────────────────────────
                triggered = self.sequencer.update(result.label)
                if triggered:
                    self.effect_state.clear()
                    # Keep _seq_match_idx at len(sequence) — all green during
                    # announcement. Reset happens when announcement ends (below).

                # Detect announcement → effect transition and reset strip
                currently_announcing = self.sequencer.is_announcing
                if self._was_announcing and not currently_announcing and self.sequencer.active_jutsu:
                    self._seq_match_idx  = 0
                    self._last_seq_label = None
                self._was_announcing = currently_announcing

                # ── OpenCV drawing ────────────────────────────────────────────
                frame = self._draw_detection(frame, result)
                frame = self._apply_ghost(frame)
                frame = self._draw_jutsu_sequence(frame)
                frame = self._draw_jutsu_announcement(frame)   # ← name banner

                # ── Visual effect (only after announcement ends) ───────────────
                if self.sequencer.active_jutsu and not self.sequencer.is_announcing:
                    frame = apply_effect(
                        frame,
                        self.sequencer.active_jutsu,
                        self.sequencer.effect_progress,
                        self.sequencer.effect_frame,
                        result.hand_bbox,
                        self.effect_state,
                    )

                frame = self._draw_rec_indicator(frame)

                if self._recording and self._video_writer is not None:
                    self._video_writer.write(frame)

                dpg.set_value("tex_camera", bgr_to_dpg_flat(frame))
                self._resize_to_viewport()
                self._update_ui(result)

                dpg.render_dearpygui_frame()

                if dpg.is_key_down(dpg.mvKey_Q):
                    break
        finally:
            if self._video_writer is not None:
                self._video_writer.release()
            self.cap.release()
            dpg.destroy_context()