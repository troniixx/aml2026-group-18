import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from collections import deque, Counter
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import torch.nn as nn

# ── config ────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT   = "Mert/mobilenetv2_best.pth"
MODEL_PATH   = "Mert/hand_landmarker.task"
CAMERA_INDEX = 1

SEAL_CLASSES = ["bird", "boar", "dog", "dragon", "hare", "horse",
                "monkey", "ox", "ram", "rat", "snake", "tiger"]

JUTSU_COMBOS = {
    ("snake", "ram", "monkey", "boar", "horse", "tiger"): "Fire Style: Fireball Jutsu",
    ("ox", "snake", "dog", "tiger"):                      "Fire Style: Dragon Flame Jutsu",
    ("rat", "tiger", "dog", "ox", "ram", "monkey"):       "Fire Style: Phoenix Flower Jutsu",
    ("ox", "monkey", "rat", "ram", "snake", "dragon"):    "Water Style: Hidden Mist Jutsu",
    ("tiger", "ox", "monkey", "hare", "ox"):              "Water Style: Water Prison Jutsu",
    ("ox", "hare", "monkey"):                             "Lightning Style: Chidori",
    ("tiger", "ox", "dog", "hare", "bird"):               "Wind Style: Great Breakthrough",
    ("tiger", "hare", "boar", "dog"):                     "Earth Style: Earth Wall",
    ("tiger", "hare", "snake"):                           "Earth Style: Rock Pillar Spears",
    ("ram", "snake", "tiger"):                            "Shadow Clone Jutsu",
    ("boar", "dog", "bird", "monkey", "ram"):             "Summoning Jutsu",
    ("tiger", "snake", "dog", "dragon"):                  "Reanimation Jutsu",
    ("ram", "horse", "snake"):                            "Body Flicker Jutsu",
    ("dog", "dragon", "bird"):                            "Transformation Jutsu",
    ("boar", "ram", "rat", "ox", "tiger"):                "Four Crimson Ray Formation",
}

CONFIDENCE_THRESHOLD = 0.3
SMOOTHING_WINDOW     = 5
# ─────────────────────────────────────────────────────────────────────────────

# ── build model ───────────────────────────────────────────────────────────────
def build_model(num_classes):
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes)
    )
    return model

model = build_model(num_classes=len(SEAL_CLASSES))
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE).eval()
print(f"Model loaded from {CHECKPOINT}")

# ── mediapipe landmarker ──────────────────────────────────────────────────────
if not Path(MODEL_PATH).exists():
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Done.")

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
landmarker = vision.HandLandmarker.create_from_options(options)
print("Landmarker ready")

# ── transforms ───────────────────────────────────────────────────────────────
infer_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── live feed ─────────────────────────────────────────────────────────────────
frame_preds = deque(maxlen=SMOOTHING_WINDOW)
conf        = 0.0
cap         = cv2.VideoCapture(CAMERA_INDEX)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(img_mp)

    if result.hand_landmarks:
        H, W  = rgb.shape[:2]
        lm    = result.hand_landmarks[0]
        xs    = [p.x * W for p in lm]
        ys    = [p.y * H for p in lm]
        pad_x = (max(xs) - min(xs)) * 0.2
        pad_y = (max(ys) - min(ys)) * 0.2
        x1    = max(0, int(min(xs) - pad_x))
        y1    = max(0, int(min(ys) - pad_y))
        x2    = min(W, int(max(xs) + pad_x))
        y2    = min(H, int(max(ys) + pad_y))
        crop  = rgb[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        tensor = infer_transforms(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits         = model(tensor)
            probs          = F.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)
            conf           = conf.item()
            pred_idx       = pred_idx.item()

        if conf >= CONFIDENCE_THRESHOLD:
            frame_preds.append(pred_idx)
    else:
        conf = 0.0
        cv2.putText(frame, "No hand detected", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if frame_preds:
        smoothed_idx   = Counter(frame_preds).most_common(1)[0][0]
        smoothed_label = SEAL_CLASSES[smoothed_idx]
    else:
        smoothed_label = "..."

    seal_sequence = [SEAL_CLASSES[i] for i in frame_preds]
    jutsu_label   = None
    for combo, name in JUTSU_COMBOS.items():
        if len(seal_sequence) >= len(combo):
            if tuple(seal_sequence[-len(combo):]) == combo:
                jutsu_label = name

    display = jutsu_label if jutsu_label else smoothed_label
    color   = (0, 255, 100) if jutsu_label else (0, 200, 255)

    cv2.putText(frame, display,             (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.putText(frame, f"conf: {conf:.2f}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Naruto Handsign — MobileNetV2", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()