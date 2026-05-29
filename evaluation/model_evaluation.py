import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from metrics import evaluate_model
from model_defs import build_mobilenet, build_swin, SEAL_CLASSES

DATA_DIR = "/Users/cynthiaj/Desktop/Advanced ML/project/aml2026-group-18/data/final/test_cropped"   # validation/test dataset
BATCH_SIZE = 32

MODEL_CONFIGS = {
    "MobileNetV2": {
        "builder": build_mobilenet,
        "checkpoint": PROJECT_ROOT / "Mert" / "mobilenetv2_best.pth"
    },
    "Swin-S": {
        "builder": build_swin,
        "checkpoint": PROJECT_ROOT / "Mert" / "swin_best.pth"
    }
}

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

test_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=eval_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Dataset classes:", test_dataset.classes)
print("Expected classes:", SEAL_CLASSES)

if test_dataset.classes != SEAL_CLASSES:
    raise ValueError(
        "Class order mismatch! "
        "ImageFolder class order is different from SEAL_CLASSES."
    )

for model_name, config in MODEL_CONFIGS.items():
    print(f"\n===== Evaluating {model_name} =====")

    model = config["builder"](num_classes=len(SEAL_CLASSES))

    checkpoint = torch.load(config["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    top1, top3, cm = evaluate_model(model, test_loader, device)

    print(f"{model_name} Top-1:", top1)
    print(f"{model_name} Top-3:", top3)
    print(f"{model_name} Confusion matrix shape:", cm.shape)
    print(cm)

