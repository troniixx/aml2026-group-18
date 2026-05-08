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
from ui.textures import bgr_to_dpg_flat, load_thumbnail


# Jutsu combos (mirrors what's in the detector)
JUTSU_COMBOS = {
    "Fire Style: Fireball Jutsu":        ("snake", "ram", "monkey", "boar", "horse", "tiger"),
    "Fire Style: Dragon Flame Jutsu":    ("ox", "snake", "dog", "tiger"),
    "Fire Style: Phoenix Flower Jutsu":  ("rat", "tiger", "dog", "ox", "ram", "monkey"),
    "Water Style: Hidden Mist Jutsu":    ("ox", "monkey", "rat", "ram", "snake", "dragon"),
    "Water Style: Water Prison Jutsu":   ("tiger", "ox", "monkey", "hare", "ox"),
    "Lightning Style: Chidori":          ("ox", "hare", "monkey"),
    "Wind Style: Great Breakthrough":    ("tiger", "ox", "dog", "hare", "bird"),
    "Earth Style: Earth Wall":           ("tiger", "hare", "boar", "dog"),
    "Earth Style: Rock Pillar Spears":   ("tiger", "hare", "snake"),
    "Shadow Clone Jutsu":                ("ram", "snake", "tiger"),
    "Summoning Jutsu":                   ("boar", "dog", "bird", "monkey", "ram"),
    "Reanimation Jutsu":                 ("tiger", "snake", "dog", "dragon"),
    "Body Flicker Jutsu":                ("ram", "horse", "snake"),
    "Transformation Jutsu":              ("dog", "dragon", "bird"),
    "Four Crimson Ray Formation":        ("boar", "ram", "rat", "ox", "tiger"),
}

# Width of the rightmost cheatsheet panel
CHEATSHEET_PANEL_WIDTH = 260
INFO_PANEL_WIDTH = 220

# Height reserved at the bottom of the camera panel for recording controls
RECORDING_BAR_HEIGHT = 38

# ── Record button color — customize here ─────────────────────────────────────
# Values are (R, G, B) in 0-255 range
RECORD_BTN_COLOR        = (176, 48, 48)   # normal state
RECORD_BTN_COLOR_HOVER  = (212, 47, 47)   # on hover
RECORD_BTN_COLOR_ACTIVE = (224, 2, 2)   # on click
# ─────────────────────────────────────────────────────────────────────────────

# How many frames between blink toggles (lower = faster blink)
BLINK_INTERVAL = 10


class App:
    """
    Encapsulates the entire DearPyGui application.

    Usage
    -----
    app = App(cfg, detector)
    app.run()
    """

    def __init__(self, cfg: Config, detector: Detector) -> None:
        self.cfg      = cfg
        self.detector = detector

        self.cap: cv2.VideoCapture = cv2.VideoCapture(cfg.ui.camera_index)
        ret, probe = self.cap.read()
        if not ret:
            raise RuntimeError("Could not open camera.")
        self.frame_h, self.frame_w = probe.shape[:2]

        # Ghost overlay state
        self._ghost_img:  Optional[np.ndarray] = None
        self._show_ghost: bool                  = False

        # Jutsu sequence overlay state
        self._jutsu_sequence: Optional[tuple] = None
        self._jutsu_name:     Optional[str]   = None

        # CV2 thumbnail cache
        self._seal_cv2_thumbs: dict[str, np.ndarray] = {}

        # Recording state
        self._recording:      bool                      = False
        self._video_writer:   Optional[cv2.VideoWriter] = None
        self._recording_path: Optional[Path]            = None

        # Blinking REC indicator state
        self._blink_frame:   int  = 0
        self._blink_visible: bool = True

        # Colours (shorthand)
        self.C_PRIMARY   = tuple(cfg.ui.primary_color)
        self.C_SECONDARY = tuple(cfg.ui.secondary_color)
        self.C_DIM       = (100, 100, 100)
        self.C_WHITE     = (220, 220, 220)
        self.C_BG_DARK   = ( 15,  15,  15)
        self.C_BG_MID    = ( 25,  25,  25)
        self.C_RED       = (200,  60,  60)

        # Seal history
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
            height=self.frame_h + 30,
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

        # Record button — color comes from the RECORD_BTN_COLOR constants above
        with dpg.theme() as self._theme_record:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,
                                    RECORD_BTN_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,
                                    RECORD_BTN_COLOR_HOVER)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,
                                    RECORD_BTN_COLOR_ACTIVE)

        # Stop button — fixed red
        with dpg.theme() as self._theme_red:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (130, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (160, 40, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (180, 50, 50))

        # Save button — blue
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

                # ── PANEL 1: camera feed + recording bar ──────────────────────
                with dpg.child_window(tag="child_left",
                                      width=fw, height=fh,
                                      no_scrollbar=True, border=False):

                    dpg.add_image("tex_camera",
                                  width=fw,
                                  height=fh - RECORDING_BAR_HEIGHT,
                                  tag="img_feed")

                    dpg.add_separator()

                    # Recording controls
                    with dpg.group(horizontal=True, tag="grp_rec_controls"):

                        # ● REC — the filled circle renders reliably on all fonts
                        dpg.add_button(
                            label="  ● REC  ",
                            tag="btn_record",
                            callback=self._start_recording,
                        )
                        dpg.bind_item_theme("btn_record", self._theme_record)

                        # Stop (hidden until recording)
                        dpg.add_button(
                            label="  stop  ",
                            tag="btn_stop",
                            callback=self._stop_recording,
                        )
                        dpg.bind_item_theme("btn_stop", self._theme_red)
                        dpg.hide_item("btn_stop")

                        # Open folder (hidden until a recording exists)
                        dpg.add_button(
                            label="  open recordings folder  ",
                            tag="btn_save",
                            callback=self._open_recordings_folder,
                        )
                        dpg.bind_item_theme("btn_save", self._theme_save)
                        dpg.hide_item("btn_save")

                # ── PANEL 2: detection info ───────────────────────────────────
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
                    dpg.add_combo(
                        items=model_names,
                        default_value=model_names[active_idx],
                        width=-1,
                        tag="combo_model",
                        callback=self._on_model_switch,
                        user_data=model_keys,
                    )
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
                    SEAL_HISTORY = 5
                    for i in range(SEAL_HISTORY):
                        dpg.add_text("", tag=f"lbl_hist_{i}")

                    dpg.add_spacer(height=8)
                    dpg.add_separator()
                    dpg.add_text("Press  Q  to quit", color=self.C_DIM)

                # ── PANEL 3: cheatsheets ──────────────────────────────────────
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

                        for jutsu_name, combo in JUTSU_COMBOS.items():
                            dpg.add_button(
                                label=jutsu_name,
                                width=-1,
                                callback=self._on_jutsu_selected,
                                user_data=(jutsu_name, combo),
                            )
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
        self._video_writer = cv2.VideoWriter(
            str(self._recording_path), fourcc, fps,
            (self.frame_w, self.frame_h),
        )
        self._recording      = True
        self._blink_frame    = 0
        self._blink_visible  = True

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

    # ──────────────────────────────────────────────────────────────────────────
    #  Blinking REC indicator drawn onto the camera frame
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_rec_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Draw a blinking red circle + REC text in the top-left of the frame."""
        if not self._recording:
            return frame

        # Advance blink counter
        self._blink_frame += 1
        if self._blink_frame >= BLINK_INTERVAL:
            self._blink_frame   = 0
            self._blink_visible = not self._blink_visible

        if not self._blink_visible:
            return frame

        # Position and size
        cx, cy = 22, 22      # circle centre
        r      = 10          # radius
        margin = 6           # gap between circle and text

        # Filled red circle
        cv2.circle(frame, (cx, cy), r, (0, 0, 220), -1)

        # Thin dark border so it shows on bright backgrounds
        cv2.circle(frame, (cx, cy), r, (0, 0, 0), 1)

        # "REC" label to the right of the circle
        cv2.putText(
            frame, "REC",
            (cx + r + margin, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA,
        )

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    #  Ghost overlay  (activating clears jutsu sequence)
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
    #  Jutsu sequence overlay  (activating clears ghost)
    # ──────────────────────────────────────────────────────────────────────────

    def _on_jutsu_selected(self, sender, app_data, user_data) -> None:
        self._show_ghost = False
        self._jutsu_name, self._jutsu_sequence = user_data

    def _clear_jutsu_sequence(self) -> None:
        self._jutsu_sequence = None
        self._jutsu_name     = None

    def _draw_jutsu_sequence(self, frame: np.ndarray) -> np.ndarray:
        if not self._jutsu_sequence:
            return frame

        fh, fw     = frame.shape[:2]
        n          = len(self._jutsu_sequence)
        thumb_size = min(64, (fw - 20) // max(n, 1))
        label_h    = 18
        bar_h      = thumb_size + label_h + 20
        margin_x   = 10
        margin_y   = 10
        bar_y      = fh - bar_h - margin_y

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y - 6), (fw, fh), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        prim_bgr = (self.C_PRIMARY[2], self.C_PRIMARY[1], self.C_PRIMARY[0])
        cv2.putText(frame, self._jutsu_name or "",
                    (margin_x, bar_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, prim_bgr, 1, cv2.LINE_AA)

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

            cv2.rectangle(frame, (x - 1, y - 1),
                          (x + thumb_size + 1, y + thumb_size + 1), prim_bgr, 1)

            if i < n - 1:
                ax = x + thumb_size + 2
                ay = y + thumb_size // 2
                cv2.arrowedLine(frame, (ax, ay), (ax + 4, ay),
                                (160, 160, 160), 1, tipLength=0.5)

            label = seal.capitalize()
            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.putText(frame, label,
                        (x + (thumb_size - lw) // 2, y + thumb_size + label_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    #  UI updates
    # ──────────────────────────────────────────────────────────────────────────

    def _on_model_switch(self, sender, app_data, user_data) -> None:
        model_keys  = user_data
        model_names = [m.name for m in self.cfg.models.values()]
        key = model_keys[model_names.index(app_data)]
        self.detector.switch_model(key)
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
            self._last_seal = result.label

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

    def _resize_to_viewport(self) -> None:
        vp_w = dpg.get_viewport_client_width()
        vp_h = dpg.get_viewport_client_height()

        left_w = max(1, vp_w - INFO_PANEL_WIDTH - CHEATSHEET_PANEL_WIDTH - 2)
        left_h = max(1, vp_h)

        img_area_h = max(1, left_h - RECORDING_BAR_HEIGHT)

        aspect = self.frame_w / self.frame_h
        if left_w / img_area_h > aspect:
            img_h = img_area_h
            img_w = int(img_h * aspect)
        else:
            img_w = left_w
            img_h = int(img_w / aspect)

        dpg.configure_item("child_left",       width=left_w,                    height=left_h)
        dpg.configure_item("child_right",      width=INFO_PANEL_WIDTH,   height=left_h)
        dpg.configure_item("child_cheatsheet", width=CHEATSHEET_PANEL_WIDTH,    height=left_h)
        dpg.configure_item("img_feed",         width=img_w,                     height=img_h)

    def _draw_detection(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
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

                result = self.detector.predict(frame)

                frame = self._draw_detection(frame, result)
                frame = self._apply_ghost(frame)
                frame = self._draw_jutsu_sequence(frame)
                frame = self._draw_rec_indicator(frame)   # ← blinking REC dot

                # Write frame to video (after all overlays are drawn)
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