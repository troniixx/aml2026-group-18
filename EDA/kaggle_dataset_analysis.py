from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import numpy as np
import sys


class Tee:
    def __init__(self, file_path: Path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # write to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def analyze_dataset(root: Path) -> pd.DataFrame:
    records = []
    for split in ["train", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found")
            continue
        for seal_dir in sorted(split_dir.iterdir()):
            if not seal_dir.is_dir():
                continue
            images = list(seal_dir.glob("*.jpg")) + list(seal_dir.glob("*.png"))

            h, w = None, None
            if images:
                img = cv2.imread(str(images[0]))
                if img is not None:
                    h, w = img.shape[:2]

            records.append({
                "split":      split,
                "seal":       seal_dir.name,
                "n_images":   len(images),
                "img_height": h,
                "img_width":  w,
            })
    return pd.DataFrame(records)


def report(df: pd.DataFrame):
    print("=" * 55)
    print("DATASET OVERVIEW")
    print("=" * 55)

    for split, group in df.groupby("split"):
        total = group["n_images"].sum()
        print(f"\n[{split.upper()}]  total images: {total}")
        print(f"  {'Seal':<20} {'Count':>6}  {'Resolution'}")
        print(f"  {'-'*20}  {'-'*6}  {'-'*15}")
        for _, row in group.iterrows():
            res = f"{int(row.img_width)}x{int(row.img_height)}" if row.img_width else "?"
            print(f"  {row.seal:<20} {row.n_images:>6}  {res}")

        print(f"\n  Min: {group.n_images.min()}  "
              f"Max: {group.n_images.max()}  "
              f"Mean: {group.n_images.mean():.0f}  "
              f"Std: {group.n_images.std():.0f}")

        imbalance = group.n_images.max() / group.n_images.min()
        print(f"  Imbalance ratio (max/min): {imbalance:.2f}x")

    print("\n\n[TRAIN/TEST RATIO PER CLASS]")
    pivot = df.pivot(index="seal", columns="split", values="n_images")
    pivot["ratio"] = pivot["train"] / pivot["test"]
    print(pivot.to_string())


def plot(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    colors = {"train": "#4C72B0", "test": "#DD8452"}
    seals  = sorted(df["seal"].unique())
    x      = np.arange(len(seals))
    width  = 0.35

    ax1 = fig.add_subplot(gs[0])
    for i, (split, grp) in enumerate(df.groupby("split")):
        counts = [grp.loc[grp.seal == s, "n_images"].values[0] if s in grp.seal.values else 0 for s in seals]
        ax1.bar(x + i * width, counts, width, label=split, color=colors[split])
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(seals, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Image count")
    ax1.set_title("Class distribution — train vs test")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1])
    pivot = df.pivot(index="seal", columns="split", values="n_images")
    ratios = pivot["train"] / pivot["test"]
    ax2.barh(seals, ratios.loc[seals], color="#55A868")
    ax2.axvline(ratios.mean(), color="red", linestyle="--", label=f"mean ({ratios.mean():.1f}x)")
    ax2.set_xlabel("Train / Test ratio")
    ax2.set_title("Train-to-test ratio per class")
    ax2.legend(fontsize=8)

    plt.suptitle("Kaggle Naruto Hand Seal Dataset Analysis", fontweight="bold")

    img_path = out_dir / "dataset_analysis.png"
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved to: {img_path}")


if __name__ == "__main__":
    ROOT     = Path("datasets/kaggle") 
    OUT_DIR  = Path("EDA/analysis_outputs")  
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tee = Tee(OUT_DIR / "dataset_analysis.txt")
    sys.stdout = tee

    try:
        df = analyze_dataset(ROOT)
        report(df)
        plot(df, OUT_DIR)
    finally:
        sys.stdout = tee.terminal  # always restore stdout
        tee.close()
        print(f"Log saved to: {OUT_DIR / 'dataset_analysis.txt'}")