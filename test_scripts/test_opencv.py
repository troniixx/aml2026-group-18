import cv2
import numpy as np
from collections import deque

# ── CONFIG ───────────────────────────────────────────────────────────────────
PANEL_WIDTH     = 300          # width of the right-side UI panel
SEAL_HISTORY    = 5            # how many recent seals to show
CAMERA_INDEX    = 0
# ─────────────────────────────────────────────────────────────────────────────

# Dummy state — will come from your model later
prediction      = "rat"
confidence      = 0.94
jutsu_triggered = "Fire Ball Jutsu"
seal_history    = deque(["bird", "ram", "rat", "dog", "rat"], maxlen=SEAL_HISTORY)

def draw_main_feed(frame, prediction, confidence):
    """Draw bounding box and label on the camera feed."""
    h, w = frame.shape[:2]
    x1, y1 = w // 4, h // 4
    x2, y2 = 3 * w // 4, 3 * h // 4

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 150), 2)

    # Label background pill
    label      = f"{prediction}  {confidence:.0%}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1 - lh - 16), (x1 + lw + 10, y1), (0, 255, 150), -1)
    cv2.putText(frame, label,
                org=(x1 + 5, y1 - 6),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 0),
                thickness=2)
    return frame


def draw_panel(panel_h, prediction, confidence, seal_history, jutsu):
    """Draw the right-side info panel."""
    panel = np.zeros((panel_h, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)   # dark background

    y = 0  # cursor

    # ── Header ───────────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 50), (30, 30, 30), -1)
    cv2.putText(panel, "NARUTO DETECTOR",
                org=(12, 33),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.6,
                color=(0, 200, 255),
                thickness=1)
    y = 70

    # ── Current Seal ─────────────────────────────────────────────────────────
    cv2.putText(panel, "CURRENT SEAL",
                org=(12, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.45,
                color=(120, 120, 120),
                thickness=1)
    y += 8
    cv2.line(panel, (12, y), (PANEL_WIDTH - 12, y), (50, 50, 50), 1)
    y += 30

    cv2.putText(panel, prediction.upper(),
                org=(12, y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1.1,
                color=(0, 255, 150),
                thickness=2)
    y += 30

    # Confidence bar
    bar_w    = PANEL_WIDTH - 24
    filled_w = int(bar_w * confidence)
    bar_color = (0, 255, 150) if confidence > 0.75 else (0, 200, 255) if confidence > 0.5 else (0, 100, 255)
    cv2.rectangle(panel, (12, y), (12 + bar_w, y + 12), (50, 50, 50), -1)
    cv2.rectangle(panel, (12, y), (12 + filled_w, y + 12), bar_color, -1)
    cv2.putText(panel, f"{confidence:.0%}",
                org=(12 + bar_w + 4, y + 11),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(180, 180, 180),
                thickness=1)
    y += 40

    # ── Seal History ─────────────────────────────────────────────────────────
    cv2.putText(panel, "SEAL HISTORY",
                org=(12, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.45,
                color=(120, 120, 120),
                thickness=1)
    y += 8
    cv2.line(panel, (12, y), (PANEL_WIDTH - 12, y), (50, 50, 50), 1)
    y += 25

    for i, seal in enumerate(reversed(seal_history)):
        alpha  = 255 - i * 40          # fade older entries
        color  = (alpha, alpha, alpha)
        prefix = "▶ " if i == 0 else f"  "
        cv2.putText(panel, f"{prefix}{seal}",
                    org=(20, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.55 if i == 0 else 0.45,
                    color=(0, 255, 150) if i == 0 else color,
                    thickness=2 if i == 0 else 1)
        y += 28

    y += 10

    # ── Jutsu Triggered ──────────────────────────────────────────────────────
    if jutsu:
        cv2.putText(panel, "JUTSU DETECTED",
                    org=(12, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.45,
                    color=(120, 120, 120),
                    thickness=1)
        y += 8
        cv2.line(panel, (12, y), (PANEL_WIDTH - 12, y), (50, 50, 50), 1)
        y += 30

        cv2.rectangle(panel, (8, y - 20), (PANEL_WIDTH - 8, y + 20), (0, 60, 120), -1)
        cv2.rectangle(panel, (8, y - 20), (PANEL_WIDTH - 8, y + 20), (0, 100, 255), 1)
        cv2.putText(panel, jutsu.upper(),
                    org=(14, y + 8),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.5,
                    color=(0, 180, 255),
                    thickness=1)
        y += 50

    # ── Footer / Controls ────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, panel_h - 35), (PANEL_WIDTH, panel_h), (30, 30, 30), -1)
    cv2.putText(panel, "press Q to quit",
                org=(12, panel_h - 12),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(80, 80, 80),
                thickness=1)

    return panel


cap = cv2.VideoCapture(CAMERA_INDEX)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── Fix mirrored camera ───────────────────────────────────────────────────
    frame = cv2.flip(frame, 1)

    frame = draw_main_feed(frame, prediction, confidence)
    panel = draw_panel(frame.shape[0], prediction, confidence, seal_history, jutsu_triggered)

    # Stitch feed and panel side by side
    display = np.hstack([frame, panel])

    cv2.imshow("Naruto Hand Seal Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()