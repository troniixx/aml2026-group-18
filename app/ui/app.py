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


# Jutsu combos live here for the cheatsheet (mirrors what's in the detector)
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

        # ── Jutsu sequence overlay state ──────────────────────────────────────
        self._jutsu_sequence: Optional[tuple]   = None   # e.g. ("snake", "ram", ...)
        self._jutsu_name:     Optional[str]     = None   # e.g. "Fireball Jutsu"

        # CV2 thumbnail cache for drawing the sequence on the frame
        self._seal_cv2_thumbs: dict[str, np.ndarray] = {}

        # Colours (shorthand)
        self.C_PRIMARY   = tuple(cfg.ui.primary_color)
        self.C_SECONDARY = tuple(cfg.ui.secondary_color)
        self.C_DIM       = (100, 100, 100)
        self.C_WHITE     = (220, 220, 220)
        self.C_BG_DARK   = ( 15,  15,  15)
        self.C_BG_MID    = ( 25,  25,  25)

        # Seal history — only records when the seal changes
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
            width=self.frame_w + self.cfg.ui.panel_width,
            height=self.frame_h + 30,
            resizable=True,
            min_width=640,
            min_height=360,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("win_main", True)

    def _load_fonts(self) -> None:
        """Load a larger font for the seal label. Falls back gracefully."""
        self._font_large = None
        # Try common Windows system fonts
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",   # Arial Bold
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibrib.ttf",  # Calibri Bold
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

    def _register_textures(self, probe: np.ndarray) -> None:
        size = self.cfg.ui.seal_thumb_size
        with dpg.texture_registry():
            # Live camera feed (dynamic)
            dpg.add_dynamic_texture(
                self.frame_w, self.frame_h,
                bgr_to_dpg_flat(probe),
                tag="tex_camera",
            )

            # Logo
            logo_path = logo_path = Path(__file__).parent.parent.parent / "assets" / "Naruto_hand_seal_detector_title.png"
            logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
            if logo is None:
                raise FileNotFoundError(f"Logo not found at: {logo_path.resolve()}")
            if len(logo.shape) == 3 and logo.shape[2] == 4:
                alpha = logo[:, :, 3:4] / 255.0
                bg    = np.full_like(logo[:, :, :3], 25)
                logo  = (logo[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
            target_width  = self.cfg.ui.panel_width - 16
            orig_h, orig_w = logo.shape[:2]
            target_height = int(target_width * orig_h / orig_w)
            logo = cv2.resize(logo, (target_width, target_height))
            dpg.add_static_texture(
                logo.shape[1], logo.shape[0],
                bgr_to_dpg_flat(logo),
                tag="tex_logo",
            )

            # Seal thumbnails (static, loaded once)
            # Also keep a CV2 copy for drawing the sequence on the camera frame
            for idx, seal in enumerate(self.cfg.all_seals):
                thumb = load_thumbnail(
                    self.cfg.seal_images.get(seal),
                    size, seal, idx,
                )
                self._seal_cv2_thumbs[seal] = thumb.copy()   # ← store CV2 copy
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

                # ── RIGHT: info panel ─────────────────────────────────────────
                with dpg.child_window(tag="child_right",
                                      width=pw, height=fh,
                                      no_scrollbar=False, border=False):

                    # Logo + active model
                    with dpg.group(horizontal=True):
                        dpg.add_image("tex_logo")
                    dpg.add_text("", tag="lbl_active_model", color=self.C_DIM)
                    dpg.add_separator()
                    dpg.add_spacer(height=2)

                    # Model selector
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

                    # ── Current seal (larger) + confidence inline ─────────────
                    dpg.add_text("CURRENT SEAL", color=self.C_PRIMARY)
                    dpg.add_spacer(height=2)
                    # Seal name and confidence on the same row
                    with dpg.group(horizontal=True):
                        dpg.add_text(" — ", tag="lbl_seal", color=self.C_SECONDARY)
                        dpg.add_text("",  tag="lbl_conf_pct", color=self.C_DIM)
                    # Bind large font to seal label if available
                    if self._font_large is not None:
                        dpg.bind_item_font("lbl_seal", self._font_large)
                        dpg.bind_item_font("lbl_conf_pct", self._font_large)

                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    # Seal history
                    dpg.add_text("SEAL HISTORY", color=self.C_PRIMARY)
                    dpg.add_spacer(height=2)
                    SEAL_HISTORY = 5
                    for i in range(SEAL_HISTORY):
                        dpg.add_text("", tag=f"lbl_hist_{i}")
                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    # ── Seal cheatsheet (collapsible) ─────────────────────────
                    with dpg.collapsing_header(label="Seal Cheatsheet",
                                               default_open=False):
                        dpg.add_spacer(height=1)
                        dpg.add_text("Click a seal to overlay it on the feed",
                                     color=self.C_DIM)
                        dpg.add_spacer(height=2)

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

                        dpg.add_spacer(height=4)
                        dpg.add_button(label="Clear overlay", width=-1,
                                       callback=lambda: self._clear_ghost())

                    dpg.add_spacer(height=2)
                    dpg.add_separator()
                    dpg.add_spacer(height=2)

                    # ── Jutsu cheatsheet (collapsible) ────────────────────────
                    with dpg.collapsing_header(label="Jutsu Cheatsheet",
                                               default_open=False):
                        dpg.add_spacer(height=1)
                        dpg.add_text("Click a jutsu to see its hand sign sequence\non the camera feed",
                                     color=self.C_DIM)
                        dpg.add_spacer(height=2)

                        # One button per jutsu — clicking previews the sequence
                        for jutsu_name, combo in JUTSU_COMBOS.items():
                            dpg.add_button(
                                label=jutsu_name,
                                width=-1,
                                callback=self._on_jutsu_selected,
                                user_data=(jutsu_name, combo),
                            )
                            dpg.add_spacer(height=2)

                        dpg.add_spacer(height=2)
                        dpg.add_button(label="Clear sequence", width=-1,
                                       callback=lambda: self._clear_jutsu_sequence())

                    dpg.add_spacer(height=2)
                    dpg.add_separator()
                    dpg.add_text("Press  Q  to quit", color=self.C_DIM)

    # ──────────────────────────────────────────────────────────────────────────
    #  Ghost overlay
    # ──────────────────────────────────────────────────────────────────────────

    def _load_ghost(self, seal_name: str) -> None:
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
    #  Jutsu sequence overlay
    # ──────────────────────────────────────────────────────────────────────────

    def _on_jutsu_selected(self, sender, app_data, user_data) -> None:
        """Called when the user clicks a jutsu in the cheatsheet."""
        self._jutsu_name, self._jutsu_sequence = user_data

    def _clear_jutsu_sequence(self) -> None:
        self._jutsu_sequence = None
        self._jutsu_name     = None

    def _draw_jutsu_sequence(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the selected jutsu's hand-sign sequence as a row of thumbnails
        along the bottom of the camera frame.
        """
        if not self._jutsu_sequence:
            return frame

        fh, fw     = frame.shape[:2]
        n          = len(self._jutsu_sequence)
        thumb_size = min(64, (fw - 20) // max(n, 1))   # fit all seals in one row
        label_h    = 18                                  # px reserved for text below each thumb
        bar_h      = thumb_size + label_h + 20          # total strip height
        margin_x   = 10
        margin_y   = 10
        bar_y      = fh - bar_h - margin_y

        # Semi-transparent dark background strip
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (0, bar_y - 6),
                      (fw, fh),
                      (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Jutsu name at the top of the strip
        name_text = self._jutsu_name or ""
        prim_bgr  = (self.C_PRIMARY[2], self.C_PRIMARY[1], self.C_PRIMARY[0])
        cv2.putText(frame, name_text,
                    (margin_x, bar_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, prim_bgr, 1, cv2.LINE_AA)

        # Draw each seal thumbnail + name
        total_w = n * thumb_size + (n - 1) * 6
        start_x = (fw - total_w) // 2   # centre the row

        for i, seal in enumerate(self._jutsu_sequence):
            x = start_x + i * (thumb_size + 6)
            y = bar_y + 14

            # Thumbnail
            if seal in self._seal_cv2_thumbs:
                thumb = cv2.resize(self._seal_cv2_thumbs[seal], (thumb_size, thumb_size))
                frame[y:y + thumb_size, x:x + thumb_size] = thumb
            else:
                # Fallback: coloured rectangle
                cv2.rectangle(frame, (x, y), (x + thumb_size, y + thumb_size),
                              (60, 60, 60), -1)

            # Border
            cv2.rectangle(frame, (x - 1, y - 1),
                          (x + thumb_size + 1, y + thumb_size + 1),
                          prim_bgr, 1)

            # Arrow between seals
            if i < n - 1:
                ax = x + thumb_size + 2
                ay = y + thumb_size // 2
                cv2.arrowedLine(frame, (ax, ay), (ax + 4, ay),
                                (160, 160, 160), 1, tipLength=0.5)

            # Seal name below thumbnail
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
        """Called when the user selects a different model in the combo box."""
        model_keys  = user_data
        model_names = [m.name for m in self.cfg.models.values()]
        key = model_keys[model_names.index(app_data)]
        self.detector.switch_model(key)
        dpg.set_value("lbl_active_model",
                      f"Active: {self.cfg.models[key].name}")

    def _update_ui(self, result: DetectionResult) -> None:
        """Push a DetectionResult into all DPG widgets."""
        SEAL_HISTORY = 5

        label_text = result.label.upper() if result.label else "—"
        conf_text  = f"  {result.confidence:.0%}" if result.label else ""

        dpg.set_value("lbl_seal",     label_text)
        dpg.set_value("lbl_conf_pct", conf_text)

        # Only record a new entry when the detected seal actually changes
        if result.label and result.label != self._last_seal:
            self._seal_history.insert(0, result.label)
            self._seal_history = self._seal_history[:SEAL_HISTORY]
            self._last_seal = result.label

        # Display history
        for i in range(SEAL_HISTORY):
            if i < len(self._seal_history):
                seal   = self._seal_history[i]
                prefix = "▶  " if i == 0 else "     "
                fade   = max(60, 220 - i * 40)
                color  = self.C_PRIMARY if i == 0 else (fade, fade, fade)
                dpg.set_value(f"lbl_hist_{i}",      f"{prefix}{seal.capitalize()}")
                dpg.configure_item(f"lbl_hist_{i}", color=color)
            else:
                # Clear slots that haven't been filled yet
                dpg.set_value(f"lbl_hist_{i}", "")

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
        """Draw the model's bounding box + label on the frame."""
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
                frame = self._draw_jutsu_sequence(frame)   # ← sequence strip

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