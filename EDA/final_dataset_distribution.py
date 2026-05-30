"""Analyzes the Final dataset and creates a stacked bar chart by class, split, and image source."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


OWN_NAMES = {"ishana", "mert", "jeni", "ajeong", "chenxi"}


def get_source(filename: str) -> str:
    name = filename.lower()
    if name.startswith("img"):
        return "kaggle"
    if any(person in name for person in OWN_NAMES):
        return "own"
    return "roboflow"


def count_images(split_dir: Path) -> dict[str, dict[str, int]]:
    """
    Returns {class_name: {"kaggle": n, "own": n, "roboflow": n}}
    """
    counts = {}
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower()
        counts[class_name] = {"kaggle": 0, "own": 0, "roboflow": 0}
        for img in class_dir.iterdir():
            if img.is_file():
                counts[class_name][get_source(img.name)] += 1
    return counts


def plot(
    train_counts: dict,
    test_counts:  dict,
    output_path:  Path,
):
    classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
    n       = len(classes)
    x       = np.arange(n)

    # Colors
    colors = {
        "kaggle":   "#4C72B0",
        "own":      "#BA58C4",
        "roboflow": "#DD8452",
    }

    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, (split_counts, offset, label_suffix) in enumerate([
        (train_counts, -bar_w / 2, "Train"),
        (test_counts,  +bar_w / 2, "Test"),
    ]):
        bottoms = np.zeros(n)
        for source in ("kaggle", "own", "roboflow"):
            values = np.array([split_counts.get(cls, {}).get(source, 0) for cls in classes])
            bars   = ax.bar(
                x + offset, values, bar_w,
                bottom=bottoms,
                color=colors[source],
                alpha=0.9 if label_suffix == "Train" else 0.5,
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += values

        # Total label on top of each bar
        totals = np.array([sum(split_counts.get(cls, {}).values()) for cls in classes])
        for xi, total in zip(x + offset, totals):
            if total > 0:
                ax.text(xi, total + 2, str(total),
                        ha="center", va="bottom", fontsize=7, color="#333333")

    # ── Legend ────────────────────────────────────────────────────────────────
    source_patches = [
        mpatches.Patch(color=colors["kaggle"],   label="Kaggle"),
        mpatches.Patch(color=colors["own"],       label="Self-recorded"),
        mpatches.Patch(color=colors["roboflow"],  label="Roboflow"),
    ]
    alpha_patches = [
        mpatches.Patch(color="grey", alpha=0.9, label="Train (full opacity)"),
        mpatches.Patch(color="grey", alpha=0.5, label="Test (50% opacity)"),
    ]
    ax.legend(handles=source_patches + alpha_patches,
              loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Image count")
    ax.set_title("Final Dataset — Class Distribution by Source and Split", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to: {output_path.resolve()}")


def print_summary(train_counts: dict, test_counts: dict):
    classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
    header  = f"{'Class':<12} {'Train':>7} {'Test':>7}  {'Kaggle(tr)':>10} {'Own(tr)':>8} {'Robo(tr)':>8}"
    print("\n" + "=" * len(header))
    print("FINAL DATASET SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for cls in classes:
        tr   = train_counts.get(cls, {})
        te   = test_counts.get(cls,  {})
        t_total = sum(tr.values())
        e_total = sum(te.values())
        print(f"  {cls:<12} {t_total:>7} {e_total:>7}  "
              f"{tr.get('kaggle',0):>10} {tr.get('own',0):>8} {tr.get('roboflow',0):>8}")
    print("-" * len(header))
    all_tr = sum(sum(v.values()) for v in train_counts.values())
    all_te = sum(sum(v.values()) for v in test_counts.values())
    print(f"  {'TOTAL':<12} {all_tr:>7} {all_te:>7}")


if __name__ == "__main__":
    REPO_ROOT     = Path(__file__).resolve().parent.parent
    FINAL_DATASET = FINAL_DATASET = REPO_ROOT / "datasets" / "final"
    OUTPUT_PATH   = OUTPUT_PATH   = Path(__file__).resolve().parent / "final_dataset_distribution.png"

    train_dir = FINAL_DATASET / "train"
    test_dir  = FINAL_DATASET / "test"

    print("Counting train images...")
    train_counts = count_images(train_dir)

    print("Counting test images...")
    test_counts = count_images(test_dir)

    print_summary(train_counts, test_counts)
    plot(train_counts, test_counts, OUTPUT_PATH)