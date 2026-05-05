"""
core/detector.py

Owns all inference logic shared between models:
  MediaPipe hand crop → torchvision transform → PyTorch model → smooth → combo check

The model architecture (build_model function) is imported dynamically from each
{model}_live.py script. Only the build function is used from those files —
all the live-feed loop code is ignored.

REQUIRED CHANGE TO EACH _live.py
─────────────────────────────────
Wrap all module-level execution code in  if __name__ == "__main__":
so that importing the file only defines build_model() without running
the camera loop. See the comment block below for exactly what to guard.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import importlib.util
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from torchvision import transforms

from core.config import Config, ModelConfig


# ── Shared inference constants (same in both live scripts) ───────────────────
SMOOTHING_WINDOW = 5
COMBO_WINDOW     = 5.0   # seconds — how long a seal sequence stays alive
MEDIAPIPE_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Shared torchvision transform (same in both live scripts) ─────────────────
INFER_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Output type ───────────────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    label:         Optional[str]    # predicted seal name, or None if below threshold
    confidence:    float            # 0.0 – 1.0
    jutsu:         Optional[str]    # jutsu name if a combo just fired, else None
    history:       list[str]        # recent seals, most-recent first
    hand_bbox:     Optional[tuple]  # (x1,y1,x2,y2) of detected hand, or None


class Detector:
    """
    Unified detector that works with any model defined by a build_model() function
    and a .pth weights file.

    Parameters
    ----------
    cfg : loaded Config object
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg         = cfg
        self._active_key = cfg.active_model
        self._model      = None
        self._conf_threshold: float = 0.5

        # Per-frame smoothing state
        self._frame_preds: deque  = deque(maxlen=SMOOTHING_WINDOW)
        self._seal_sequence: list = []   # [(seal_name, timestamp), ...]
        self._last_seal:     Optional[str]   = None
        self._jutsu_label:   Optional[str]   = None
        self._jutsu_until:   float           = 0.0

        self._seal_classes: list[str] = []  # set from live script on model load

        self._setup_device()
        self._setup_mediapipe(cfg.mediapipe_model)
        self._load_model(cfg.active_model_config)

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> DetectionResult:
        """
        Run the full pipeline on a single BGR frame and return a DetectionResult.
        Safe to call every frame — returns a no-hand result gracefully.
        """
        label, conf, bbox = self._run_inference(frame)
        self._update_sequence(label, conf)
        jutsu = self._check_jutsu()

        # Build history list (most-recent first, unique consecutive)
        history = [s for s, _ in self._seal_sequence][::-1]

        return DetectionResult(
            label=label,
            confidence=conf,
            jutsu=jutsu,
            history=history,
            hand_bbox=bbox,
        )

    def switch_model(self, model_key: str) -> None:
        """Hot-swap the active model at runtime."""
        if model_key == self._active_key:
            return
        if model_key not in self.cfg.models:
            raise ValueError(f"Unknown model key: {model_key!r}")
        print(f"[Detector] Switching: {self._active_key} → {model_key}")
        self._active_key = model_key
        self._frame_preds.clear()
        self._load_model(self.cfg.models[model_key])

    @property
    def active_model_name(self) -> str:
        return self.cfg.models[self._active_key].name

    # ──────────────────────────────────────────────────────────────────────────
    #  Setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_device(self) -> None:
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        print(f"[Detector] Using device: {self._device}")

    def _setup_mediapipe(self, model_path: Optional[Path]) -> None:
        """Initialise the MediaPipe hand landmarker (downloaded if missing)."""
        if model_path is None:
            model_path = Path("hand_landmarker.task")

        if not model_path.exists():
            print(f"[Detector] Downloading MediaPipe hand landmarker → {model_path}")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MEDIAPIPE_URL, model_path)
            print("[Detector] Download complete.")

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        print("[Detector] MediaPipe hand landmarker ready.")

    # ──────────────────────────────────────────────────────────────────────────
    #  Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self, model_cfg: ModelConfig) -> None:
        """
        Import build_model() from {model}_live.py, build the architecture,
        then load the .pth weights.

        The live script must have all execution code (model loading, camera loop)
        wrapped in  if __name__ == "__main__":  so that importing it only
        defines the build_model / build_swin function.
        """
        print(f"[Detector] Loading '{model_cfg.name}' from {model_cfg.weights_path}")

        # ── Dynamically import the build function from the live script ─────────
        spec   = importlib.util.spec_from_file_location("_live", model_cfg.script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Support both naming conventions: build_model (mobilenet) / build_swin (swin)
        build_fn = getattr(module, "build_model", None) or getattr(module, "build_swin", None)
        if build_fn is None:
            raise AttributeError(
                f"Could not find 'build_model' or 'build_swin' in {model_cfg.script_path}. "
                "Make sure the function is defined at module level."
            )

        # ── Pull SEAL_CLASSES from the live script (must match training order) ──
        self._seal_classes = getattr(module, "SEAL_CLASSES", None)
        if self._seal_classes is None:
            raise AttributeError(
                f"Could not find 'SEAL_CLASSES' in {model_cfg.script_path}. "
                "Make sure it is defined at module level outside __main__."
            )
        print(f"[Detector] Class order: {self._seal_classes}")

        # ── Build architecture & load weights ─────────────────────────────────
        self._model = build_fn(num_classes=len(self._seal_classes))
        self._model.load_state_dict(
            torch.load(str(model_cfg.weights_path), map_location=self._device)
        )
        self._model.to(self._device).eval()

        self._conf_threshold = model_cfg.confidence_threshold
        print(f"[Detector] Ready. Confidence threshold: {self._conf_threshold}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Inference pipeline (identical logic to both live scripts)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_inference(
        self, frame: np.ndarray
    ) -> tuple[str, float, Optional[tuple]]:
        """
        1. Run MediaPipe to detect the hand and get a bounding box crop.
        2. Run the PyTorch model on the crop.
        3. Apply confidence threshold + smoothing.
        Returns (label, confidence, bbox_or_None).
        """
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(img_mp)

        if not result.hand_landmarks:
            self._frame_preds.clear()
            return None, 0.0, None

        # ── Crop around hand landmarks ─────────────────────────────────────────
        H, W = rgb.shape[:2]
        lm   = result.hand_landmarks[0]
        xs   = [p.x * W for p in lm]
        ys   = [p.y * H for p in lm]
        pad_x = (max(xs) - min(xs)) * 0.2
        pad_y = (max(ys) - min(ys)) * 0.2
        x1 = max(0, int(min(xs) - pad_x))
        y1 = max(0, int(min(ys) - pad_y))
        x2 = min(W, int(max(xs) + pad_x))
        y2 = min(H, int(max(ys) + pad_y))
        crop = rgb[y1:y2, x1:x2]

        if crop.size == 0:
            return None, 0.0, (x1, y1, x2, y2)

        # ── Model inference ────────────────────────────────────────────────────
        tensor = INFER_TRANSFORMS(crop).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits         = self._model(tensor)
            probs          = F.softmax(logits, dim=1)[0]   # shape: (num_classes,)

        # Sort descending to get top-2
        top2_probs, top2_idx = probs.topk(2)
        pred_idx = top2_idx[0].item()

        # Margin = top1 - top2: near 0 when uncertain, near 1 when clearly one seal.
        # Much more honest than raw softmax probability.
        conf = (top2_probs[0] - top2_probs[1]).item()

        # ── Confidence threshold + smoothing ──────────────────────────────────
        if conf >= self._conf_threshold:
            self._frame_preds.append(pred_idx)
        else:
            self._frame_preds.clear()

        if self._frame_preds:
            smoothed_idx = Counter(self._frame_preds).most_common(1)[0][0]
            label = self._seal_classes[smoothed_idx]
        else:
            label = None
            conf  = 0.0

        return label, conf, (x1, y1, x2, y2)

    # ──────────────────────────────────────────────────────────────────────────
    #  Seal sequence + jutsu combo (same logic as both live scripts)
    # ──────────────────────────────────────────────────────────────────────────

    def _update_sequence(self, label: str, conf: float) -> None:
        """Append to the timed seal sequence when a new seal is confidently held."""
        now = time.time()

        # Expire old seals
        self._seal_sequence = [
            (s, t) for s, t in self._seal_sequence if now - t < COMBO_WINDOW
        ]

        # Add new seal only when label is confident and has changed
        if label is not None and conf >= self._conf_threshold and label != self._last_seal:
            self._seal_sequence.append((label, now))
            self._last_seal = label

    def _check_jutsu(self) -> Optional[str]:
        """Check whether the current sequence tail matches a known combo."""
        now           = time.time()
        current_seals = tuple(s for s, _ in self._seal_sequence)

        for combo_tuple, name in self.cfg.jutsu_combos.items():
            combo = tuple(combo_tuple)
            n     = len(combo)
            if len(current_seals) >= n and current_seals[-n:] == combo:
                self._jutsu_label  = name
                self._jutsu_until  = now + 3.0
                self._seal_sequence = []          # reset after trigger
                self._frame_preds.clear()
                self._last_seal = None
                return name

        # Keep showing jutsu for 3 s after trigger
        if self._jutsu_label and now > self._jutsu_until:
            self._jutsu_label = None

        return self._jutsu_label