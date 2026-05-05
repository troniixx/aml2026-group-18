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

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from core.config import Config
from core.detector import Detector, DetectionResult
from ui.textures import bgr_to_dpg_flat, load_thumbnail


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

        # Colours (shorthand)
        self.C_PRIMARY   = tuple(cfg.ui.primary_color)
        self.C_SECONDARY = tuple(cfg.ui.secondary_color)
        self.C_DIM       = (100, 100, 100)
        self.C_WHITE     = (220, 220, 220)
        self.C_BG_DARK   = ( 15,  15,  15)
        self.C_BG_MID    = ( 25,  25,  25)

        self._setup_dpg(probe)

    # ──────────────────────────────────────────────────────────────────────────
    #  Setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_dpg(self, probe: np.ndarray) -> None:
        dpg.create_context()
        self._build_theme()
        self._register_textures(probe)
        self._build_ui()

        dpg.create_viewport(
            title="Naruto Hand Seal Detection",
            width=self.frame_w + self.cfg.ui.panel_width,
            height=self.frame_h + 30,
            resizable=True,
            min_width=640,
            min_height=360,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("win_main", True)

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

    def _register_textures(self, probe: np.ndarray) -> None:
        size = self.cfg.ui.seal_thumb_size
        with dpg.texture_registry():
            # Live camera feed (dynamic)
            dpg.add_dynamic_texture(
                self.frame_w, self.frame_h,
                bgr_to_dpg_flat(probe),
                tag="tex_camera",
            )
            # Seal thumbnails (static, loaded once)
            for idx, seal in enumerate(self.cfg.all_seals):
                thumb = load_thumbnail(
                    self.cfg.seal_images.get(seal),
                    size, seal, idx,
                )
                dpg.add_static_texture(
                    size, size,
                    bgr_to_dpg_flat(thumb),
                    tag=f"tex_seal_{seal}",
                )

    def _build_ui(self) -> None:
        fw, fh = self.frame_w, self.frame_h
        pw     = self.cfg.ui.panel_width

        with dpg.window(tag="win_main", no_title_bar=True, no_resize=True,
                        no_move=True, no_scrollbar=True,
                        width=fw + pw, height=fh, pos=(0, 0)):

            with dpg.group(horizontal=True):

                # ── LEFT: camera feed ─────────────────────────────────────────
                with dpg.child_window(tag="child_left",
                                      width=fw, height=fh,
                                      no_scrollbar=True, border=False):
                    dpg.add_image("tex_camera", width=fw, height=fh,
                                  tag="img_feed")
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text("▶  DETECTED:", color=self.C_DIM)
                        dpg.add_text("—", tag="lbl_feed_seal",
                                     color=self.C_PRIMARY)

                # ── RIGHT: info panel ─────────────────────────────────────────
                with dpg.child_window(tag="child_right",
                                      width=pw, height=fh,
                                      no_scrollbar=False, border=False):

                    # Header row: title + active model indicator
                    with dpg.group(horizontal=True):
                        dpg.add_text("NARUTO  DETECTOR", color=self.C_SECONDARY)
                    dpg.add_text("", tag="lbl_active_model", color=self.C_DIM)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    # Model selector
                    dpg.add_text("MODEL", color=self.C_DIM)
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
                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=8)

                    # Current seal
                    dpg.add_text("CURRENT SEAL", color=self.C_DIM)
                    dpg.add_text("—", tag="lbl_seal", color=self.C_PRIMARY)
                    dpg.add_spacer(height=4)
                    dpg.add_text("Confidence", color=self.C_DIM)
                    dpg.add_progress_bar(tag="bar_conf",
                                         default_value=0.0,
                                         width=-1, height=14)
                    dpg.add_text("—", tag="lbl_conf_pct", color=self.C_DIM)
                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=8)

                    # Seal history
                    dpg.add_text("SEAL HISTORY", color=self.C_DIM)
                    dpg.add_spacer(height=4)
                    for i in range(self.cfg.ui.panel_width):  # over-allocate; only show SEAL_HISTORY
                        pass
                    SEAL_HISTORY = 5
                    for i in range(SEAL_HISTORY):
                        dpg.add_text("", tag=f"lbl_hist_{i}")
                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=8)

                    # Jutsu detected
                    dpg.add_text("JUTSU DETECTED", color=self.C_DIM)
                    dpg.add_spacer(height=4)
                    with dpg.child_window(tag="box_jutsu", width=-1, height=54,
                                          border=True, no_scrollbar=True):
                        dpg.add_text("—", tag="lbl_jutsu",
                                     color=self.C_SECONDARY)
                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=8)

                    # Seal cheatsheet (collapsible)
                    with dpg.collapsing_header(label="Seal Cheatsheet",
                                               default_open=False):
                        dpg.add_spacer(height=4)
                        dpg.add_text("Click a seal to overlay it on the feed",
                                     color=self.C_DIM)
                        dpg.add_spacer(height=6)

                        COLS   = 3
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
                                            dpg.add_text(seal.capitalize(),
                                                         color=self.C_DIM)

                        dpg.add_spacer(height=6)
                        dpg.add_button(label="Clear overlay", width=-1,
                                       callback=lambda: self._clear_ghost())

                    dpg.add_spacer(height=8)
                    dpg.add_separator()
                    dpg.add_text("Press  Q  to quit", color=self.C_DIM)

    # ──────────────────────────────────────────────────────────────────────────
    #  Ghost overlay
    # ──────────────────────────────────────────────────────────────────────────

    def _load_ghost(self, seal_name: str) -> None:
        path = (self.cfg.ghost_images.get(seal_name)
                or self.cfg.seal_images.get(seal_name))
        if path:
            # Read with alpha channel if present (IMREAD_UNCHANGED keeps BGRA)
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            img = raw if raw is not None else self._make_pip_placeholder(seal_name)
        else:
            img = self._make_pip_placeholder(seal_name)
        # Store at original resolution — _apply_ghost resizes to pip_size at draw time
        self._ghost_img  = img
        self._show_ghost = True

    def _make_pip_placeholder(self, seal_name: str) -> np.ndarray:
        """Small labelled square used when no image file is available."""
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
        """Paste the selected seal image as a PiP square in the bottom-right corner."""
        if not self._show_ghost or self._ghost_img is None:
            return frame

        fh, fw = frame.shape[:2]

        # PiP size: 20% of the shorter frame dimension, clamped to a sensible range
        pip_size = max(80, min(200, int(min(fh, fw) * 0.20)))
        margin   = 12   # px gap from the frame edge

        thumb = cv2.resize(self._ghost_img, (pip_size, pip_size))

        # Destination region in the bottom-right corner
        x1 = fw - pip_size - margin
        y1 = fh - pip_size - margin
        x2 = x1 + pip_size
        y2 = y1 + pip_size

        # ── Handle transparent PNG (BGRA) ────────────────────────────────────
        if thumb.shape[2] == 4 if len(thumb.shape) == 3 and thumb.shape[2] == 4 else False:
            bgr   = thumb[:, :, :3]
            alpha = thumb[:, :, 3:4].astype(np.float32) / 255.0
            roi   = frame[y1:y2, x1:x2].astype(np.float32)
            frame[y1:y2, x1:x2] = (bgr * alpha + roi * (1 - alpha)).astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = thumb

        # ── Border around the PiP ─────────────────────────────────────────────
        border_color = (self.C_PRIMARY[2], self.C_PRIMARY[1], self.C_PRIMARY[0])
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), border_color, 2)

        return frame



    # ──────────────────────────────────────────────────────────────────────────
    #  UI updates
    # ──────────────────────────────────────────────────────────────────────────

    def _on_model_switch(self, sender, app_data, user_data) -> None:
        """Called when the user selects a different model in the combo box."""
        model_keys  = user_data                           # list of keys
        model_names = [m.name for m in self.cfg.models.values()]
        key = model_keys[model_names.index(app_data)]
        self.detector.switch_model(key)
        dpg.set_value("lbl_active_model",
                      f"Active: {self.cfg.models[key].name}")

    def _update_ui(self, result: DetectionResult) -> None:
        """Push a DetectionResult into all DPG widgets."""
        label_text = result.label.upper() if result.label else "—"
        dpg.set_value("lbl_seal",       label_text)
        dpg.set_value("lbl_feed_seal",  label_text)
        dpg.set_value("bar_conf",       result.confidence)
        dpg.set_value("lbl_conf_pct",   f"{result.confidence:.0%}" if result.label else "—")

        SEAL_HISTORY = 5
        for i, seal in enumerate(result.history[:SEAL_HISTORY]):
            prefix = "▶  " if i == 0 else "     "
            fade   = max(60, 220 - i * 40)
            color  = self.C_PRIMARY if i == 0 else (fade, fade, fade)
            dpg.set_value(f"lbl_hist_{i}",      f"{prefix}{seal.capitalize()}")
            dpg.configure_item(f"lbl_hist_{i}", color=color)

        dpg.set_value("lbl_jutsu",
                      result.jutsu.upper() if result.jutsu else "—")

    def _resize_to_viewport(self) -> None:
        vp_w = dpg.get_viewport_client_width()
        vp_h = dpg.get_viewport_client_height()

        left_w = max(1, vp_w - self.cfg.ui.panel_width - 2)
        left_h = max(1, vp_h)

        aspect = self.frame_w / self.frame_h
        if left_w / left_h > aspect:
            img_h = left_h
            img_w = int(img_h * aspect)
        else:
            img_w = left_w
            img_h = int(img_w / aspect)

        dpg.configure_item("child_left",  width=left_w,                    height=left_h)
        dpg.configure_item("child_right", width=self.cfg.ui.panel_width,   height=left_h)
        dpg.configure_item("img_feed",    width=img_w,                     height=img_h)

    def _draw_detection(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw the model's bounding box + label on the frame (OpenCV layer).
        Does nothing if no hand was detected this frame."""
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
        """Start the render loop. Blocks until the window is closed or Q pressed."""
        try:
            while dpg.is_dearpygui_running():
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # ── Model inference ───────────────────────────────────────────
                result = self.detector.predict(frame)

                # ── OpenCV drawing layer ──────────────────────────────────────
                frame = self._draw_detection(frame, result)
                frame = self._apply_ghost(frame)

                # ── Push to DPG ───────────────────────────────────────────────
                dpg.set_value("tex_camera", bgr_to_dpg_flat(frame))
                self._resize_to_viewport()
                self._update_ui(result)

                dpg.render_dearpygui_frame()

                if dpg.is_key_down(dpg.mvKey_Q):
                    break
        finally:
            self.cap.release()
            dpg.destroy_context()