"""Counts the current state of own curated dataset => who has uploaded what"""

from pathlib import Path


def analyze_state(folder_with_data: Path, seals: list[str], people: list[str], report_path: Path):
    
    # ── Parse uploaded files ──────────────────────────────────────────────────
    uploaded = {seal: {person: False for person in people} for seal in seals}
    unrecognized = []

    for f in Path(folder_with_data).glob("*.mp4"):
        parts = f.stem.split("_")
        if len(parts) < 3:
            unrecognized.append(f.name)
            continue

        seal   = parts[0].lower()
        person = parts[1].lower()

        if seal not in seals:
            unrecognized.append(f.name)
            continue
        if person not in people:
            unrecognized.append(f.name)
            continue

        uploaded[seal][person] = True

    # ── Build table ───────────────────────────────────────────────────────────
    col_w      = 10
    seal_col_w = 10
    tick       = "✓"
    cross      = "·"

    header     = f"{'Seal':<{seal_col_w}}" + "".join(f"{p:^{col_w}}" for p in people) + f"{'Total':^{col_w}}"
    divider    = "-" * len(header)

    lines = []
    lines.append("OWN DATA UPLOAD TRACKER")
    lines.append("=" * len(header))
    lines.append("")
    lines.append(header)
    lines.append(divider)

    person_totals = {p: 0 for p in people}
    seal_totals   = {}

    for seal in seals:
        row_count = 0
        row = f"{seal:<{seal_col_w}}"
        for person in people:
            has_uploaded = uploaded[seal][person]
            row += f"{tick if has_uploaded else cross:^{col_w}}"
            if has_uploaded:
                person_totals[person] += 1
                row_count += 1
        row += f"{row_count:^{col_w}}"
        seal_totals[seal] = row_count
        lines.append(row)

    lines.append(divider)

    # Totals row
    totals_row = f"{'Total':<{seal_col_w}}"
    for person in people:
        totals_row += f"{person_totals[person]:^{col_w}}"
    totals_row += f"{sum(person_totals.values()):^{col_w}}"
    lines.append(totals_row)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_possible = len(seals) * len(people)
    total_uploaded = sum(person_totals.values())

    lines.append("")
    lines.append("─" * len(header))
    lines.append("SUMMARY")
    lines.append("─" * len(header))
    lines.append(f"  Total uploaded : {total_uploaded} / {total_possible}")
    lines.append(f"  Progress       : {total_uploaded / total_possible * 100:.1f}%")
    lines.append("")
    lines.append("  Per person:")
    for person in people:
        bar_filled = int((person_totals[person] / len(seals)) * 20)
        bar        = "█" * bar_filled + "░" * (20 - bar_filled)
        lines.append(f"    {person:<10} {person_totals[person]:>2} / {len(seals)}  [{bar}]")

    lines.append("")
    lines.append("  Missing uploads:")
    any_missing = False
    for seal in seals:
        missing = [p for p in people if not uploaded[seal][p]]
        if missing:
            any_missing = True
            lines.append(f"    {seal:<10} still needed from: {', '.join(missing)}")
    if not any_missing:
        lines.append("    None — all uploads complete!")

    if unrecognized:
        lines.append("")
        lines.append("  Unrecognized files (check naming convention):")
        for f in unrecognized:
            lines.append(f"    {f}")

    output = "\n".join(lines)
    print(output)

    report_path.write_text(output, encoding="utf-8")
    print(f"\nReport saved to: {report_path.resolve()}")


if __name__ == "__main__":
    seals            = ["bird", "boar", "dog", "dragon", "hare", "horse", "monkey", "ox", "ram", "rat", "snake", "tiger", "zero"]
    people           = ["chenxi", "jeni", "mert", "ishana", "ajeong"]
    folder_with_data = Path("../datasets/own_data") # "datasets/own_data"
    report_path      = Path("./reports/uploaded_data_report.txt") # "Dataset_Adder/reports/uploaded_data_report.txt"

    analyze_state(folder_with_data, seals, people, report_path)
    # run from Dataset_adder folder or adjust path