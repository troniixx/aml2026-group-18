import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights, Swin_S_Weights


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = Path("/Users/cynthiaj/Desktop/Advanced ML/project/aml2026-group-18/data/final")
TRAIN_DIR_CROPPED = DATA_ROOT / "train_cropped"

SEAL_CLASSES = [
    "bird", "boar", "dog", "dragon", "hare", "horse",
    "monkey", "ox", "ram", "rat", "snake", "tiger"
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(SEAL_CLASSES)}
NUM_CLASSES = len(SEAL_CLASSES)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 32

# 
EPOCHS_PHASE_1 = 2
EPOCHS_PHASE_2 = 3

LR = 3e-4
LR_BACKBONE = 5e-5

PATIENCE = 5
MIN_DELTA = 0.005

print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def collect_samples(split_dir: Path):
    samples = []

    for cls in SEAL_CLASSES:
        cls_dir = split_dir / cls

        if not cls_dir.is_dir():
            print(f"Warning: missing folder {cls_dir}")
            continue

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            for img_path in cls_dir.glob(ext):
                samples.append((img_path, CLASS_TO_IDX[cls]))

    return samples


class HandSealDatasetFromSamples(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self.targets = [label for _, label in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none"
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def build_fold_criterion(train_samples):
    train_labels = [label for _, label in train_samples]
    counts = Counter(train_labels)
    total = sum(counts.values())

    weights = torch.tensor(
        [
            total / (NUM_CLASSES * max(counts.get(i, 1), 1))
            for i in range(NUM_CLASSES)
        ],
        dtype=torch.float
    ).to(DEVICE)

    return FocalLoss(weight=weights, gamma=2.0)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def build_mobilenet(num_classes=12):
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    for p in model.features.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.8),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes)
    )

    return model


def build_swin(num_classes=12):
    model = models.swin_s(weights=Swin_S_Weights.DEFAULT)

    for p in model.parameters():
        p.requires_grad = False

    in_features = model.head.in_features

    model.head = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    return model


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_topk(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)

        top1_preds = logits.argmax(dim=1)
        correct_top1 += (top1_preds == labels).sum().item()

        top3_preds = logits.topk(k=3, dim=1).indices
        correct_top3 += top3_preds.eq(labels.view(-1, 1)).any(dim=1).sum().item()

        total += imgs.size(0)

    avg_loss = total_loss / total
    top1 = correct_top1 / total
    top3 = correct_top3 / total

    return avg_loss, top1, top3


# ---------------------------------------------------------------------
# Training one fold
# ---------------------------------------------------------------------

def train_one_fold(model_name, model, train_loader, val_loader, criterion):
    model.to(DEVICE)

    # -----------------------
    # Phase 1: train head only
    # -----------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS_PHASE_1
    )

    print(f"{model_name} Phase 1: training head only")

    for epoch in range(EPOCHS_PHASE_1):
        model.train()

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_loss, val_top1, val_top3 = evaluate_topk(
            model,
            val_loader,
            criterion
        )

        print(
            f"Phase 1 Epoch {epoch + 1:02d} | "
            f"val loss {val_loss:.3f} | "
            f"top1 {val_top1:.3f} | "
            f"top3 {val_top3:.3f}"
        )

    # -----------------------
    # Phase 2: fine-tune backbone
    # -----------------------

    if model_name == "MobileNetV2":
        for layer in list(model.features.children())[-10:]:
            for p in layer.parameters():
                p.requires_grad = True

        optimizer = torch.optim.AdamW([
            {"params": model.features.parameters(), "lr": LR_BACKBONE},
            {"params": model.classifier.parameters(), "lr": LR},
        ])

    elif model_name == "Swin-S":
        for layer in list(model.features.children())[-4:]:
            for p in layer.parameters():
                p.requires_grad = True

        optimizer = torch.optim.AdamW([
            {"params": model.features.parameters(), "lr": LR_BACKBONE},
            {"params": model.head.parameters(), "lr": LR},
        ])

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_BACKBONE, LR],
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS_PHASE_2,
        pct_start=0.3
    )

    best_top1 = 0.0
    best_top3 = 0.0
    patience_counter = 0

    print(f"{model_name} Phase 2: fine-tuning backbone")

    for epoch in range(EPOCHS_PHASE_2):
        model.train()

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        val_loss, val_top1, val_top3 = evaluate_topk(
            model,
            val_loader,
            criterion
        )

        improved = val_top1 > best_top1 + MIN_DELTA

        if improved:
            best_top1 = val_top1
            best_top3 = val_top3
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Phase 2 Epoch {epoch + 1:02d} | "
            f"val loss {val_loss:.3f} | "
            f"top1 {val_top1:.3f} | "
            f"top3 {val_top3:.3f}"
            + (" <- best" if improved else f" patience {patience_counter}/{PATIENCE}")
        )

        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

    return best_top1, best_top3


# ---------------------------------------------------------------------
# 5-fold CV
# ---------------------------------------------------------------------

def run_5fold_stability(model_name, build_model_fn, samples):
    labels = np.array([label for _, label in samples])
    indices = np.arange(len(samples))

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    fold_top1 = []
    fold_top3 = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n========== {model_name} | Fold {fold + 1}/5 ==========")

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        train_dataset = HandSealDatasetFromSamples(
            train_samples,
            transform=train_transforms
        )

        val_dataset = HandSealDatasetFromSamples(
            val_samples,
            transform=val_transforms
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        criterion = build_fold_criterion(train_samples)

        model = build_model_fn(num_classes=NUM_CLASSES)

        best_top1, best_top3 = train_one_fold(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion
        )

        fold_top1.append(best_top1)
        fold_top3.append(best_top3)

        print(
            f"{model_name} Fold {fold + 1} result | "
            f"best top1 {best_top1:.4f} | "
            f"best top3 {best_top3:.4f}"
        )

    print(f"\n========== {model_name} Stability Result ==========")
    print("Top-1 scores:", fold_top1)
    print("Top-1 mean:", np.mean(fold_top1))
    print("Top-1 std:", np.std(fold_top1))

    print("Top-3 scores:", fold_top3)
    print("Top-3 mean:", np.mean(fold_top3))
    print("Top-3 std:", np.std(fold_top3))

    return fold_top1, fold_top3

if __name__ == "__main__":
    samples = collect_samples(TRAIN_DIR_CROPPED)

    print(f"Total samples for CV: {len(samples)}")

    class_counts = Counter(label for _, label in samples)
    print("Class distribution:")
    for cls in SEAL_CLASSES:
        print(cls, class_counts[CLASS_TO_IDX[cls]])

    mobilenet_top1, mobilenet_top3 = run_5fold_stability(
        model_name="MobileNetV2",
        build_model_fn=build_mobilenet,
        samples=samples
    )

    swin_top1, swin_top3 = run_5fold_stability(
        model_name="Swin-S",
        build_model_fn=build_swin,
        samples=samples
    )