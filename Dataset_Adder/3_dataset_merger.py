"""Merges our own curated dataset with the dataset available from kaggle (checks for existing). Also creates a summary report."""

from pathlib import Path
import datetime
import shutil


def merge_kaggle(kaggle_folder: Path) -> dict[str, list[Path]]:
    """
    Merges train and test folders from Kaggle into one dict of {seal: [image_paths]}.
    Does not copy anything to disk — just collects paths.
    """
    collected = {}
    for split in ("train", "test"):
        split_dir = kaggle_folder / split
        if not split_dir.exists():
            print(f"  Warning: {split_dir} not found, skipping.")
            continue
        for seal_dir in split_dir.iterdir():
            if not seal_dir.is_dir():
                continue
            seal = seal_dir.name.lower()
            images = list(seal_dir.glob("*.jpg")) + list(seal_dir.glob("*.jpeg")) + list(seal_dir.glob("*.png"))
            collected.setdefault(seal, []).extend(images)

    total = sum(len(v) for v in collected.items())
    print(f"  Kaggle: found {sum(len(v) for v in collected.values())} images across {len(collected)} seals")
    return collected


def merge_own_kaggle(
    kaggle_collected: dict[str, list[Path]],
    own_dataset: Path,
    output_path: Path,
    report_path: Path,
):
    """
    Copies Kaggle + own curated images into output_path/{seal}/.
    Skips files that already exist in the destination.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect own data paths
    own_collected = {}
    if own_dataset.exists():
        for seal_dir in own_dataset.iterdir():
            if not seal_dir.is_dir():
                continue
            seal   = seal_dir.name.lower()
            images = list(seal_dir.glob("*.jpg")) + list(seal_dir.glob("*.jpeg")) + list(seal_dir.glob("*.png"))
            own_collected[seal] = images
    else:
        print(f"  Warning: own dataset folder {own_dataset} not found, merging Kaggle only.")

    # All seals across both sources
    all_seals = sorted(set(list(kaggle_collected.keys()) + list(own_collected.keys())))

    results = {}

    for seal in all_seals:
        seal_out = output_path / seal
        seal_out.mkdir(exist_ok=True)

        # Existing files in destination (for skip check)
        existing = {f.name for f in seal_out.glob("*")}

        counts = {"kaggle_copied": 0, "kaggle_skipped": 0, "own_copied": 0, "own_skipped": 0}

        # ── Copy Kaggle images ────────────────────────────────────────────────
        for img in kaggle_collected.get(seal, []):
            dest_name = f"kaggle_{img.name}"
            if dest_name in existing:
                counts["kaggle_skipped"] += 1
            else:
                shutil.copy2(img, seal_out / dest_name)
                existing.add(dest_name)
                counts["kaggle_copied"] += 1

        # ── Copy own images ───────────────────────────────────────────────────
        for img in own_collected.get(seal, []):
            dest_name = f"own_{img.name}"
            if dest_name in existing:
                counts["own_skipped"] += 1
            else:
                shutil.copy2(img, seal_out / dest_name)
                existing.add(dest_name)
                counts["own_copied"] += 1

        counts["total"] = len(list(seal_out.glob("*")))
        results[seal]   = counts

    build_report(results, kaggle_collected, own_collected, output_path, report_path)


def build_report(
    results:           dict,
    kaggle_collected:  dict,
    own_collected:     dict,
    output_path:       Path,
    report_path:       Path,
):
    total_copied  = sum(r["kaggle_copied"] + r["own_copied"]   for r in results.values())
    total_skipped = sum(r["kaggle_skipped"] + r["own_skipped"] for r in results.values())
    total_on_disk = sum(r["total"]                             for r in results.values())

    lines = []
    lines.append("DATASET MERGE REPORT")
    lines.append("=" * 65)
    lines.append(f"  Output folder    : {output_path.resolve()}")
    lines.append(f"  Total copied     : {total_copied}")
    lines.append(f"  Total skipped    : {total_skipped}  (already on disk)")
    lines.append(f"  Total on disk    : {total_on_disk}")
    lines.append("")

    # Per-seal breakdown
    lines.append("─" * 65)
    lines.append(f"  {'Seal':<12} {'Kaggle':>8} {'Own':>8} {'Skipped':>8} {'Total':>8}")
    lines.append(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}")

    for seal in sorted(results.keys()):
        r = results[seal]
        kaggle_n = len(kaggle_collected.get(seal, []))
        own_n    = len(own_collected.get(seal, []))
        skipped  = r["kaggle_skipped"] + r["own_skipped"]
        lines.append(
            f"  {seal:<12} {kaggle_n:>8} {own_n:>8} {skipped:>8} {r['total']:>8}"
        )

    lines.append("")
    lines.append("─" * 65)
    lines.append("IMBALANCE CHECK")
    lines.append("─" * 65)
    totals    = {seal: results[seal]["total"] for seal in results}
    min_seal  = min(totals, key=totals.get)
    max_seal  = max(totals, key=totals.get)
    lines.append(f"  Min : {min_seal:<12} {totals[min_seal]} images")
    lines.append(f"  Max : {max_seal:<12} {totals[max_seal]} images")
    lines.append(f"  Ratio (max/min) incl. zero : {totals[max_seal] / totals[min_seal]:.2f}x")

    totals_no_zero = {seal: count for seal, count in totals.items() if seal != "zero"}
    if totals_no_zero:
        min_seal_nz = min(totals_no_zero, key=totals_no_zero.get)
        max_seal_nz = max(totals_no_zero, key=totals_no_zero.get)
        lines.append(f"  Ratio (max/min) excl. zero : {totals_no_zero[max_seal_nz] / totals_no_zero[min_seal_nz]:.2f}x")
        lines.append(f"    (min: {min_seal_nz} {totals_no_zero[min_seal_nz]}, max: {max_seal_nz} {totals_no_zero[max_seal_nz]})")

    output = "\n".join(lines)
    print(output)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(output, encoding="utf-8")
    print(f"\nReport saved to: {report_path.resolve()}")


if __name__ == "__main__":
    # run from dataset_adder folder
    raw_kaggle_dataset  = Path("../datasets/Kaggle")
    own_curated_dataset = Path("../datasets/own_data_processed")
    merged_dataset_path = Path("../datasets/MERGED")

    date_formatted = datetime.datetime.now().strftime("%d-%m-%y")
    report_path    = Path(f"./reports/dataset_merger_report_{date_formatted}_mert_ajeong_added.txt")

    print("Step 1: collecting Kaggle images...")
    kaggle_collected = merge_kaggle(raw_kaggle_dataset)

    print("\nStep 2: merging Kaggle + own data...")
    merge_own_kaggle(kaggle_collected, own_curated_dataset, merged_dataset_path, report_path)