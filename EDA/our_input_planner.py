from pathlib import Path
import re


def parse_analysis(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Analysis file not found: {path.resolve()}")

    text = path.read_text(encoding="utf-8")

    sections = {}
    for split in ("TEST", "TRAIN"):
        pattern = rf"\[{split}\].*?(?=\[|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find [{split}] section in {path}")
        sections[split] = match.group()

    row_pattern = re.compile(r"^\s{2}(\w+)\s+(\d+)\s+\d+x\d+", re.MULTILINE)

    counts: dict[str, dict[str, int]] = {}
    for split, block in sections.items():
        for match in row_pattern.finditer(block):
            seal, count = match.group(1), int(match.group(2))
            counts.setdefault(seal, {})[split.lower()] = count

    if not counts:
        raise ValueError(f"No class rows parsed from {path} — check file format.")

    merged = {}
    for seal, splits in counts.items():
        train = splits.get("train", 0)
        test  = splits.get("test",  0)
        merged[seal] = train + test

    return merged


def priority(deficit: int) -> str:
    if deficit >= 100:
        return "🔴 HIGH"
    elif deficit >= 60:
        return "🟠 MED"
    elif deficit > 0:
        return "🟡 LOW"
    else:
        return "✅ OVERSAMPLE — cap or leave"


def build_rows(counts: dict, target: int) -> list[dict]:
    rows = []
    for seal, kaggle in sorted(counts.items(), key=lambda x: x[1]):
        deficit = target - kaggle
        rows.append({
            "seal":     seal,
            "kaggle":   kaggle,
            "target":   target,
            "deficit":  deficit,
            "priority": priority(deficit),
        })
    return rows


def format_table(rows: list[dict]) -> str:
    lines = []
    lines.append(f"{'Class':<12} {'Kaggle':>8} {'Target':>8} {'Deficit':>8}  {'Priority'}")
    lines.append(f"{'-'*12}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*30}")
    for r in rows:
        lines.append(
            f"{r['seal']:<12} {r['kaggle']:>8} {r['target']:>8} {r['deficit']:>8}  {r['priority']}"
        )
    return "\n".join(lines)


def frames_per_clip(clip_length_s: int, extraction_fps: int, trim_seconds: float) -> int:
    """Usable frames extracted from one clip after trimming both ends."""
    clean_seconds = max(0, clip_length_s - 2 * trim_seconds)
    return int(clean_seconds * extraction_fps)


def format_stats(
    rows: list[dict],
    team_members: int,
    clip_length_s: int,
    extraction_fps: int,
    trim_seconds: float,
) -> str:
    needed        = [r for r in rows if r["deficit"] > 0]
    total_needed  = sum(r["deficit"] for r in needed)
    per_person    = -(-total_needed // team_members)          # ceiling division

    usable_frames = frames_per_clip(clip_length_s, extraction_fps, trim_seconds)
    clips_needed  = -(-per_person // usable_frames)           # ceiling division
    clips_per_seal = -(-clips_needed // len(rows))            # spread across all seals

    lines = []
    lines.append("─" * 50)
    lines.append("STATISTICS")
    lines.append("─" * 50)
    lines.append(f"  Target per class         : {TARGET_VALUE}")
    lines.append(f"  Classes needing images   : {len(needed)} / {len(rows)}")
    lines.append(f"  Total images needed      : {total_needed}")
    lines.append(f"  Team members             : {team_members}")
    lines.append(f"  Images per person        : ~{per_person}  (ceiling)")
    lines.append("")
    lines.append("  Breakdown by priority:")
    for label in ["🔴 HIGH", "🟠 MED", "🟡 LOW"]:
        group  = [r for r in needed if r["priority"] == label]
        subtot = sum(r["deficit"] for r in group)
        seals  = ", ".join(r["seal"] for r in group)
        lines.append(f"    {label}  →  {subtot:>4} images  ({seals})")

    lines.append("")
    lines.append("─" * 50)
    lines.append("VIDEO RECORDING ESTIMATE")
    lines.append("─" * 50)
    lines.append(f"  Clip length              : {clip_length_s}s")
    lines.append(f"  Extraction FPS           : {extraction_fps}fps")
    lines.append(f"  Trim (each end)          : {trim_seconds}s")
    lines.append(f"  Usable frames per clip   : ~{usable_frames}")
    lines.append(f"  Clips needed per person  : ~{clips_needed}  (to cover ~{per_person} images)")
    lines.append(f"  Spread across {len(rows)} seals     : ~{clips_per_seal} clip(s) per seal per person")
    lines.append("")
    lines.append(f"  Tip: record {clips_per_seal + 1} clips per seal per person for a small safety margin.")

    return "\n".join(lines)


if __name__ == "__main__":

    TARGET_VALUE    = 200
    TEAM_MEMBERS    = 5
    CLIP_LENGTH_S   = 8        # seconds per recorded clip
    EXTRACTION_FPS  = 5        # frames extracted per second
    TRIM_SECONDS    = 2.0      # seconds trimmed from each end of the clip
    ANALYSIS_PATH   = Path("EDA/analysis_outputs/dataset_analysis.txt")
    OUTPUT_PATH     = Path(f"EDA/analysis_outputs/TODO_target_{TARGET_VALUE}.txt")

    print(f"Reading analysis from: {ANALYSIS_PATH.resolve()}")
    kaggle_counts = parse_analysis(ANALYSIS_PATH)
    print(f"Parsed {len(kaggle_counts)} classes: {', '.join(sorted(kaggle_counts))}\n")

    rows  = build_rows(kaggle_counts, TARGET_VALUE)
    table = format_table(rows)
    stats = format_stats(rows, TEAM_MEMBERS, CLIP_LENGTH_S, EXTRACTION_FPS, TRIM_SECONDS)

    output = "\n".join([
        "TARGET ANALYSIS — Naruto Hand Seal Dataset",
        "=" * 50,
        "",
        table,
        "",
        stats,
    ])

    print(output)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"\nSaved to: {OUTPUT_PATH.resolve()}")