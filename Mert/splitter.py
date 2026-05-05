import shutil
import random
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
SRC_DIR   = Path("/Users/merterol/Desktop/aml2026-group-18/datasets/own_data_processed")
TRAIN_DIR = Path("/Users/merterol/Desktop/aml2026-group-18/datasets/final/train")
TEST_DIR  = Path("/Users/merterol/Desktop/aml2026-group-18/datasets/final/test")
TEST_RATIO = 0.2
RANDOM_SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

if not SRC_DIR.exists():
    raise FileNotFoundError(f"Source folder not found: {SRC_DIR}")

for cls_dir in sorted(SRC_DIR.iterdir()):
    if not cls_dir.is_dir():
        continue

    cls = cls_dir.name
    images = [f for f in cls_dir.iterdir() if f.suffix in EXTENSIONS]

    if not images:
        print(f"  {cls}: no images found, skipping")
        continue

    random.shuffle(images)
    n_test  = max(1, int(len(images) * TEST_RATIO))
    test_files  = images[:n_test]
    train_files = images[n_test:]

    for split, files in [("train", train_files), ("test", test_files)]:
        dst = (TRAIN_DIR if split == "train" else TEST_DIR) / cls
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, dst / f.name)

    print(f"  {cls}: {len(train_files)} train, {len(test_files)} test")

print("\nDone.")