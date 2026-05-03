"""Converts own uploaded videos into frames to curate own dataset"""

from pathlib import Path
import cv2
import datetime


def extract_frames(
    video_path: Path,
    output_root: Path,
    extraction_fps: int,
) -> dict:
    """
    Extract frames from a single video and save as JPEGs.
    Returns a result dict for the report.
    """
    # ── Parse filename ────────────────────────────────────────────────────────
    parts = video_path.stem.split("_")
    if len(parts) < 3:
        return {"status": "skipped", "reason": "unrecognized filename format", "file": video_path.name, "frames_saved": 0}

    seal, person, distance = parts[0], parts[1], parts[2]


    # ── Check if already processed ────────────────────────────────────────────
    out_dir         = output_root / seal
    prefix          = f"{seal}_{person}_{distance}_"
    existing_frames = list(out_dir.glob(f"{prefix}*.jpeg")) if out_dir.exists() else []

    if existing_frames:
        return {
            "status":       "already_processed",
            "file":         video_path.name,
            "seal":         seal,
            "person":       person,
            "distance":     distance,
            "frames_saved": len(existing_frames),
        }
    
    # ── Prepare output directory ──────────────────────────────────────────────
    out_dir = output_root / seal
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract frames ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "reason": "could not open video", "file": video_path.name, "frames_saved": 0}

    source_fps     = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s     = total_frames / source_fps if source_fps > 0 else 0
    frame_interval = max(1, round(source_fps / extraction_fps))

    frame_idx, split_num = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_name = f"{seal}_{person}_{distance}_{split_num:04d}.jpeg"
            cv2.imwrite(str(out_dir / out_name), frame)
            split_num += 1
        frame_idx += 1

    cap.release()

    return {
        "status":       "ok",
        "file":         video_path.name,
        "seal":         seal,
        "person":       person,
        "distance":     distance,
        "duration_s":   round(duration_s, 2),
        "source_fps":   round(source_fps, 1),
        "frames_saved": split_num,
    }


def build_report(results: list[dict], extraction_fps: int, report_path: Path):
    ok       = [r for r in results if r["status"] == "ok"]
    skipped  = [r for r in results if r["status"] == "skipped"]
    errors   = [r for r in results if r["status"] == "error"]
    already   = [r for r in results if r["status"] == "already_processed"]

    # Per-person and per-seal counts
    person_counts = {}
    seal_counts   = {}
    for r in ok:
        person_counts[r["person"]] = person_counts.get(r["person"], 0) + r["frames_saved"]
        seal_counts[r["seal"]]     = seal_counts.get(r["seal"],   0) + r["frames_saved"]

    lines = []
    lines.append("FRAME EXTRACTION REPORT")
    lines.append("=" * 60)
    lines.append(f"  Extraction FPS   : {extraction_fps}fps")
    lines.append(f"  Videos processed : {len(ok)}")
    lines.append(f"  Already done     : {len(already)}  (skipped)")
    lines.append(f"  Videos skipped   : {len(skipped)}")
    lines.append(f"  Errors           : {len(errors)}")
    lines.append(f"  Total frames     : {sum(r['frames_saved'] for r in ok)}")
    lines.append("")

    # Per-video breakdown
    lines.append("─" * 60)
    lines.append("PER VIDEO")
    lines.append("─" * 60)
    lines.append(f"  {'File':<40} {'Dur':>5}s  {'Frames':>7}")
    lines.append(f"  {'-'*40}  {'-'*5}   {'-'*7}")
    for r in ok:
        lines.append(f"  {r['file']:<40} {r['duration_s']:>5}s  {r['frames_saved']:>7}")

    # Per-seal summary
    lines.append("")
    lines.append("─" * 60)
    lines.append("FRAMES PER SEAL")
    lines.append("─" * 60)
    for seal, count in sorted(seal_counts.items()):
        lines.append(f"  {seal:<12} {count:>5} frames")

    # Per-person summary
    lines.append("")
    lines.append("─" * 60)
    lines.append("FRAMES PER PERSON")
    lines.append("─" * 60)
    for person, count in sorted(person_counts.items()):
        lines.append(f"  {person:<12} {count:>5} frames")

    # Skipped / errors
    if skipped or errors:
        lines.append("")
        lines.append("─" * 60)
        lines.append("ISSUES")
        lines.append("─" * 60)
        for r in skipped + errors:
            lines.append(f"  [{r['status'].upper()}] {r['file']} — {r['reason']}")

    output = "\n".join(lines)
    print(output)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(output, encoding="utf-8")
    print(f"\nReport saved to: {report_path.resolve()}")


if __name__ == "__main__":
    video_folder        = Path("../datasets/own_data")
    output_root_folder  = Path("../datasets/own_data_processed")

    curr_date           = datetime.datetime.now()
    day                 = curr_date.strftime("%d")
    month               = curr_date.strftime("%m")
    year                = curr_date.strftime("%y")
    date_formatted      = f"{day}-{month}-{year}"
    
    report_path         = Path(f"./reports/frame_extraction_report_{date_formatted}.txt")
    extraction_fps      = 5   # frames extracted per second — yields 10–25 frames per 2–5s clip

    videos  = list(video_folder.glob("*.mp4"))
    print(f"Found {len(videos)} videos in {video_folder.resolve()}\n")

    results = [extract_frames(v, output_root_folder, extraction_fps) for v in videos]
    build_report(results, extraction_fps, report_path)