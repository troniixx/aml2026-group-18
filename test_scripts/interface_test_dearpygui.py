import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
PANEL_WIDTH     = 330       # fixed width of the right info panel (px)
SEAL_THUMB_SIZE = 64        # square size of cheatsheet thumbnails (px)
SEAL_HISTORY    = 5
CAMERA_INDEX    = 0

# ── COLOURS ───────────────────────────────────────────────────────────────────
#  Edit these two lines to retheme the entire interface.
C_SECONDARY   = (240, 83, 33)    # main accent  —  currently: naruto orange
C_PRIMARY = (240, 114, 172)     # secondary    —  currently: naruto rosa
# ─────────────────────────────────────────────────────────────────────────────
C_DIM     = (100, 100, 100)
C_WHITE   = (220, 220, 220)
C_BG_DARK = ( 15,  15,  15)
C_BG_MID  = ( 25,  25,  25)

# ── ALL SEALS ─────────────────────────────────────────────────────────────────
ALL_SEALS = ["rat", "ox", "tiger", "hare", "dragon",
             "snake", "horse", "ram", "monkey", "bird",
             "dog", "boar"]

# ── SEAL IMAGE PATHS ──────────────────────────────────────────────────────────
#  Replace None with the path to the corresponding thumbnail image file.
#  e.g.  "rat": "assets/seals/rat.png"
#  Accepted formats: anything OpenCV can read (PNG, JPG, …).
#  If None a colour-coded placeholder thumbnail is generated automatically.
SEAL_IMAGE_PATHS: dict[str, str | None] = {
    "rat": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\rat_transparent.PNG",
    "ox": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\ox_transparent.PNG",
    "tiger": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\tiger_transparent.PNG",
    "hare": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\hare_transparent.PNG",
    "dragon": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\dragon_transparent.PNG",
    "snake": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\snake_transparent.PNG",
    "horse": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\horse_transparent.PNG",
    "ram": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\ram_transparent.PNG",
    "monkey": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\monkey_transparent.PNG",
    "bird": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\bird_transparent.png",
    "dog": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\dog_transparent.PNG",
    "boar": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\boar_transparent.PNG"
}

# ── GHOST OVERLAY IMAGE PATHS ─────────────────────────────────────────────────
#  Full-frame reference image shown at 20 % opacity on the live feed.
#  Set to None to use an auto-generated placeholder.
GHOST_IMAGE_PATHS: dict[str, str | None] = {
    "rat": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\rat_transparent.PNG",
    "ox": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\ox_transparent.PNG",
    "tiger": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\tiger_transparent.PNG",
    "hare": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\hare_transparent.PNG",
    "dragon": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\dragon_transparent.PNG",
    "snake": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\snake_transparent.PNG",
    "horse": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\horse_transparent.PNG",
    "ram": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\ram_transparent.PNG",
    "monkey": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\monkey_transparent.PNG",
    "bird": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\bird_transparent.png",
    "dog": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\dog_transparent.PNG",
    "boar": r"C:\Users\babus\OneDrive\Documents\uni uzh\FS26\advanced_machine_learning\aml2026-group-18\assets\seals\transparent\boar_transparent.PNG"
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DUMMY STATE  (replace with your model outputs)
# ═══════════════════════════════════════════════════════════════════════════════
prediction      = "rat"
confidence      = 0.94
jutsu_triggered = "Fire Ball Jutsu"
seal_history    = deque(["bird", "ram", "rat", "dog", "rat"], maxlen=SEAL_HISTORY)


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA SETUP
# ═══════════════════════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(CAMERA_INDEX)
ret, _probe = cap.read()
if not ret:
    raise RuntimeError("Could not open camera.")
FRAME_H, FRAME_W = _probe.shape[:2]


# ═══════════════════════════════════════════════════════════════════════════════
#  GHOST OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════
ghost_img:  np.ndarray | None = None
show_ghost: bool              = False


def _make_placeholder_frame(seal_name: str) -> np.ndarray:
    img = np.full((FRAME_H, FRAME_W, 3), 70, dtype=np.uint8)
    cv2.putText(img, seal_name.upper(),
                (FRAME_W // 4, FRAME_H // 2),
                cv2.FONT_HERSHEY_DUPLEX, 2, (190, 190, 190), 3)
    return img


def load_ghost(seal_name: str) -> None:
    global ghost_img, show_ghost
    path = GHOST_IMAGE_PATHS.get(seal_name) or SEAL_IMAGE_PATHS.get(seal_name)
    if path:
        raw = cv2.imread(path)
        img = raw if raw is not None else _make_placeholder_frame(seal_name)
    else:
        img = _make_placeholder_frame(seal_name)
    ghost_img  = cv2.resize(img, (FRAME_W, FRAME_H))
    show_ghost = True


def clear_ghost() -> None:
    global show_ghost
    show_ghost = False


def apply_ghost(frame: np.ndarray) -> np.ndarray:
    if show_ghost and ghost_img is not None:
        return cv2.addWeighted(ghost_img, 0.50, frame, 0.80, 0)
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXTURE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def bgr_to_dpg_flat(img: np.ndarray) -> list[float]:
    """Convert BGR image → flat RGBA float32 list for DPG textures."""
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return (rgba.flatten().astype(np.float32) / 255.0).tolist()


def _make_seal_thumbnail(seal_name: str, idx: int) -> np.ndarray:
    """
    Generate a colour-coded placeholder thumbnail for *seal_name*.
    Each seal gets a unique hue derived from its list index.
    Swap out by setting SEAL_IMAGE_PATHS[seal_name] to a real file path.
    """
    hue    = int((idx / len(ALL_SEALS)) * 179)
    colour = cv2.cvtColor(np.uint8([[[hue, 160, 110]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()

    img = np.full((SEAL_THUMB_SIZE, SEAL_THUMB_SIZE, 3),
                  [max(0, c // 4) for c in colour], dtype=np.uint8)
    cv2.rectangle(img, (1, 1),
                  (SEAL_THUMB_SIZE - 2, SEAL_THUMB_SIZE - 2),
                  colour, 2)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    (tw, th), _ = cv2.getTextSize(seal_name, font, scale, thick)
    tx = (SEAL_THUMB_SIZE - tw) // 2
    ty = (SEAL_THUMB_SIZE + th) // 2
    cv2.putText(img, seal_name, (tx, ty), font, scale, (220, 220, 220), thick)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
#  DPG INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════
dpg.create_context()

# ── Theme ─────────────────────────────────────────────────────────────────────
with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       C_BG_DARK)
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        C_BG_MID)
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (35, 35, 35))
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram,  C_PRIMARY)
        dpg.add_theme_color(dpg.mvThemeCol_Text,           C_WHITE)
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

# ── Texture registry ──────────────────────────────────────────────────────────
with dpg.texture_registry():

    # Camera feed (dynamic — updated every frame)
    dpg.add_dynamic_texture(FRAME_W, FRAME_H,
                            bgr_to_dpg_flat(_probe),
                            tag="tex_camera")

    # Seal thumbnails (static — loaded once at startup)
    for _idx, _seal in enumerate(ALL_SEALS):
        _path = SEAL_IMAGE_PATHS.get(_seal)
        if _path:
            _raw = cv2.imread(_path)
            _thumb = (cv2.resize(_raw, (SEAL_THUMB_SIZE, SEAL_THUMB_SIZE))
                      if _raw is not None
                      else _make_seal_thumbnail(_seal, _idx))
        else:
            _thumb = _make_seal_thumbnail(_seal, _idx)

        dpg.add_static_texture(SEAL_THUMB_SIZE, SEAL_THUMB_SIZE,
                               bgr_to_dpg_flat(_thumb),
                               tag=f"tex_seal_{_seal}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
with dpg.window(tag="win_main", no_title_bar=True, no_resize=True,
                no_move=True, no_scrollbar=True,
                width=FRAME_W + PANEL_WIDTH, height=FRAME_H,
                pos=(0, 0)):

    with dpg.group(horizontal=True):

        # ── LEFT: camera feed ─────────────────────────────────────────────────
        with dpg.child_window(tag="child_left",
                              width=FRAME_W, height=FRAME_H,
                              no_scrollbar=True, border=False):

            dpg.add_image("tex_camera",
                          width=FRAME_W, height=FRAME_H,
                          tag="img_feed")
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_text("▶  DETECTED:", color=C_DIM)
                dpg.add_text(prediction.upper(),
                             tag="lbl_feed_seal", color=C_PRIMARY)

        # ── RIGHT: info panel ─────────────────────────────────────────────────
        with dpg.child_window(tag="child_right",
                              width=PANEL_WIDTH, height=FRAME_H,
                              no_scrollbar=False, border=False):

            # Header
            dpg.add_text("NARUTO  DETECTOR", color=C_SECONDARY)
            dpg.add_separator()
            dpg.add_spacer(height=6)

            # ── Current seal ──────────────────────────────────────────────────
            dpg.add_text("CURRENT SEAL", color=C_DIM)
            dpg.add_text(prediction.upper(), tag="lbl_seal", color=C_PRIMARY)
            dpg.add_spacer(height=4)
            dpg.add_text("Confidence", color=C_DIM)
            dpg.add_progress_bar(tag="bar_conf",
                                 default_value=confidence,
                                 width=-1, height=14)
            dpg.add_text(f"{confidence:.0%}", tag="lbl_conf_pct", color=C_DIM)
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=8)

            # ── Seal history ──────────────────────────────────────────────────
            dpg.add_text("SEAL HISTORY", color=C_DIM)
            dpg.add_spacer(height=4)
            for i in range(SEAL_HISTORY):
                dpg.add_text("", tag=f"lbl_hist_{i}")
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=8)

            # ── Jutsu detected ────────────────────────────────────────────────
            dpg.add_text("JUTSU DETECTED", color=C_DIM)
            dpg.add_spacer(height=4)
            with dpg.child_window(tag="box_jutsu", width=-1, height=54,
                                  border=True, no_scrollbar=True):
                dpg.add_text("", tag="lbl_jutsu", color=C_SECONDARY)
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=8)

            # ── Seal Cheatsheet (collapsible) ─────────────────────────────────
            with dpg.collapsing_header(label="Seal Cheatsheet",
                                       default_open=False):
                dpg.add_spacer(height=4)
                dpg.add_text("Click a seal to overlay it on the feed",
                             color=C_DIM)
                dpg.add_spacer(height=6)

                # 3-column image-button grid
                COLS   = 3
                chunks = [ALL_SEALS[i:i + COLS]
                          for i in range(0, len(ALL_SEALS), COLS)]

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
                                    # ── Swap in real images via SEAL_IMAGE_PATHS
                                    # ── The texture is already loaded above;
                                    # ── nothing here needs to change.
                                    def _cb(sender, app_data, user_data, s=seal):
                                        load_ghost(s)
                                    dpg.add_image_button(
                                        f"tex_seal_{seal}",
                                        width=SEAL_THUMB_SIZE,
                                        height=SEAL_THUMB_SIZE,
                                        callback=lambda s, a, u: load_ghost(u),
                                        user_data=seal,
                                    )
                                    dpg.add_text(seal.capitalize(),
                                                 color=C_DIM)

                dpg.add_spacer(height=6)
                dpg.add_button(label="Clear overlay", width=-1,
                               callback=clear_ghost)

            dpg.add_spacer(height=8)
            dpg.add_separator()
            dpg.add_text("Press  Q  to quit", color=C_DIM)


dpg.bind_theme(global_theme)

dpg.create_viewport(
    title="Naruto Hand Seal Detection",
    width=FRAME_W + PANEL_WIDTH,
    height=FRAME_H + 30,        # +30 for OS title bar
    resizable=True,             # ← freely resizable
    min_width=640,
    min_height=360,
)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("win_main", True)


# ═══════════════════════════════════════════════════════════════════════════════
#  UI UPDATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def update_panel() -> None:
    """Push current dummy state into all DPG widgets."""
    dpg.set_value("lbl_seal",      prediction.upper())
    dpg.set_value("lbl_feed_seal", prediction.upper())
    dpg.set_value("bar_conf",      confidence)
    dpg.set_value("lbl_conf_pct",  f"{confidence:.0%}")

    for i, seal in enumerate(reversed(seal_history)):
        prefix = "▶  " if i == 0 else "     "
        fade   = max(60, 220 - i * 40)
        color  = C_PRIMARY if i == 0 else (fade, fade, fade)
        dpg.set_value(f"lbl_hist_{i}",      f"{prefix}{seal.capitalize()}")
        dpg.configure_item(f"lbl_hist_{i}", color=color)

    dpg.set_value("lbl_jutsu",
                  jutsu_triggered.upper() if jutsu_triggered else "—")


def resize_to_viewport() -> None:
    """
    Recalculate layout every frame so the interface fills whatever size the
    user has dragged the window to.

    Layout:
      ┌─────────────────────────────┬──────────────┐
      │   camera feed  (fills rest) │  right panel │
      │   letterboxed to preserve   │  fixed width │
      │   capture aspect ratio      │              │
      └─────────────────────────────┴──────────────┘
    """
    vp_w = dpg.get_viewport_client_width()
    vp_h = dpg.get_viewport_client_height()

    left_w = max(1, vp_w - PANEL_WIDTH - 2)   # -2 for DPG internal padding
    left_h = max(1, vp_h)

    # Letterbox the camera image (preserve aspect ratio)
    aspect = FRAME_W / FRAME_H
    if left_w / left_h > aspect:
        img_h = left_h
        img_w = int(img_h * aspect)
    else:
        img_w = left_w
        img_h = int(img_w / aspect)

    dpg.configure_item("child_left",  width=left_w,    height=left_h)
    dpg.configure_item("child_right", width=PANEL_WIDTH, height=left_h)
    dpg.configure_item("img_feed",    width=img_w,     height=img_h)


def draw_bounding_box(frame: np.ndarray) -> np.ndarray:
    """Draw detection bounding box + label on the raw frame (OpenCV layer)."""
    h, w   = frame.shape[:2]
    x1, y1 = w // 4, h // 4
    x2, y2 = 3 * w // 4, 3 * h // 4

    # C_PRIMARY is RGB — OpenCV needs BGR
    bgr = (C_PRIMARY[2], C_PRIMARY[1], C_PRIMARY[0])

    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
    label = f"{prediction}  {confidence:.0%}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1 - lh - 16), (x1 + lw + 10, y1), bgr, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
while dpg.is_dearpygui_running():

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    frame = draw_bounding_box(frame)
    frame = apply_ghost(frame)

    dpg.set_value("tex_camera", bgr_to_dpg_flat(frame))

    resize_to_viewport()
    update_panel()

    dpg.render_dearpygui_frame()

    if dpg.is_key_down(dpg.mvKey_Q):
        break

cap.release()
dpg.destroy_context()