"""Splits the MERGED dataset into train, validate and test datasets."""

from pathlib import Path
import shutil
import random
import datetime
from collections import defaultdict


def get_clip_id(filename: str) -> str:
    """
    Derive a clip-level group ID from a filename so that frames from
    the same source clip always end up in the same split.

    Own files:    own_{seal}_{person}_{distance}_{frame}.jpeg
                  → group: own_{seal}_{person}_{distance}
    Kaggle files: kaggle_{anything}.jpg
                  → group: kaggle_{anything_without_last_token}
                    (Kaggle images have no clip structure; each is its own group)
    """
    stem = Path(filename).stem
    if stem.startswith("own_"):
        parts = stem.split("_")
        # drop the last token (frame number) to get the clip id
        return "_".join(parts[:-1])
    else:
        # Kaggle: treat every image as its own group (no clip structure)
        return stem


def split_dataset(
    merged_dataset: Path,
    output_dataset: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    report_path: Path,
    seed: int = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + validate + test ratios must sum to 1.0"

    random.seed(seed)

    # Wipe and recreate output folder so every run is a clean split
    if output_dataset.exists():
        shutil.rmtree(output_dataset)

    for split in ("train", "validate", "test"):
        for seal_dir in merged_dataset.iterdir():
            if seal_dir.is_dir():
                (output_dataset / split / seal_dir.name).mkdir(parents=True, exist_ok=True)

    results = {}

    for seal_dir in sorted(merged_dataset.iterdir()):
        if not seal_dir.is_dir():
            continue

        seal   = seal_dir.name
        images = sorted(seal_dir.glob("*"))

        # ── Group images by clip ID ───────────────────────────────────────────
        clip_groups: dict[str, list[Path]] = defaultdict(list)
        for img in images:
            clip_id = get_clip_id(img.name)
            clip_groups[clip_id].append(img)

        # Shuffle at the clip level, not the image level
        clip_ids = list(clip_groups.keys())
        random.shuffle(clip_ids)

        n           = len(clip_ids)
        train_end   = int(n * train_ratio)
        val_end     = train_end + int(n * val_ratio)

        split_assignment = (
            [(cid, "train")    for cid in clip_ids[:train_end]]
          + [(cid, "validate") for cid in clip_ids[train_end:val_end]]
          + [(cid, "test")     for cid in clip_ids[val_end:]]
        )

        counts = {"train": 0, "validate": 0, "test": 0, "clips": n}

        for clip_id, split in split_assignment:
            for img in clip_groups[clip_id]:
                dest = output_dataset / split / seal / img.name
                shutil.copy2(img, dest)
                counts[split] += 1

        results[seal] = counts

    build_report(results, train_ratio, val_ratio, test_ratio, output_dataset, report_path)


def build_report(
    results:      dict,
    train_ratio:  float,
    val_ratio:    float,
    test_ratio:   float,
    output_path:  Path,
    report_path:  Path,
):
    total_train = sum(r["train"]    for r in results.values())
    total_val   = sum(r["validate"] for r in results.values())
    total_test  = sum(r["test"]     for r in results.values())
    total_all   = total_train + total_val + total_test

    lines = []
    lines.append("DATASET SPLIT REPORT")
    lines.append("=" * 70)
    lines.append(f"  Output folder    : {output_path.resolve()}")
    lines.append(f"  Split ratios     : train {train_ratio:.0%}  /  validate {val_ratio:.0%}  /  test {test_ratio:.0%}")
    lines.append(f"  Total images     : {total_all}")
    lines.append(f"  Train            : {total_train}  ({total_train/total_all:.1%})")
    lines.append(f"  Validate         : {total_val}   ({total_val/total_all:.1%})")
    lines.append(f"  Test             : {total_test}  ({total_test/total_all:.1%})")
    lines.append("")

    # Per-seal breakdown
    lines.append("─" * 70)
    lines.append(f"  {'Seal':<12} {'Clips':>6} {'Train':>8} {'Validate':>10} {'Test':>8} {'Total':>8}")
    lines.append(f"  {'-'*12}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}")

    for seal in sorted(results.keys()):
        r     = results[seal]
        total = r["train"] + r["validate"] + r["test"]
        lines.append(
            f"  {seal:<12} {r['clips']:>6} {r['train']:>8} {r['validate']:>10} {r['test']:>8} {total:>8}"
        )

    lines.append("")
    lines.append("─" * 70)
    lines.append("IMBALANCE CHECK (train split)")
    lines.append("─" * 70)
    train_counts = {seal: results[seal]["train"] for seal in results}
    min_seal     = min(train_counts, key=train_counts.get)
    max_seal     = max(train_counts, key=train_counts.get)
    lines.append(f"  Min : {min_seal:<12} {train_counts[min_seal]} images")
    lines.append(f"  Max : {max_seal:<12} {train_counts[max_seal]} images")
    lines.append(f"  Ratio (max/min)  : {train_counts[max_seal] / train_counts[min_seal]:.2f}x")

    output = "\n".join(lines)
    print(output)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(output, encoding="utf-8")
    print(f"\nReport saved to: {report_path.resolve()}")


if __name__ == "__main__":
    # run from root folder:
    merged_dataset = Path("datasets/MERGED")
    output_dataset = Path("datasets/Final_Dataset")

    train    = 0.8
    validate = 0.1
    test     = 0.1

    date_formatted = datetime.datetime.now().strftime("%d-%m-%y")
    report_path    = Path(f"Dataset_Adder/reports/dataset_split_report_{date_formatted}.txt")

    split_dataset(
        merged_dataset=merged_dataset,
        output_dataset=output_dataset,
        train_ratio=train,
        val_ratio=validate,
        test_ratio=test,
        report_path=report_path,
    )