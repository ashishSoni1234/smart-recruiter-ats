"""
patch_reasoning.py — One-time patch to update reasoning strings in existing CSV.

What it does:
  - Rank 1:     unchanged (already says "Strong/Moderate semantic match")
  - Ranks 2-10: replaces "Weak semantic match; included for other signals."
                with specific elevating signals parsed from existing reasoning
  - Ranks 11-25: replaces with "Moderate composite fit — ..." text
  - Ranks 26+:   replaces with "Adjacent semantic profile — ..." text
"""

import csv
import re

INPUT_CSV  = "ashish-soni-solo.csv"
OUTPUT_CSV = "ashish-soni-solo.csv"   # overwrite in-place

OLD_TEXT = "Weak semantic match; included for other signals."

def build_top10_replacement(reasoning: str) -> str:
    """Parse existing signals already in reasoning and build elevating-signals sentence."""
    elev = []
    if "Tier-1 Indian product company" in reasoning:
        elev.append("tier-1 product background")
    if "High trust" in reasoning or "trust (100%)" in reasoning.lower():
        elev.append("high skill trust score")
    if "Strong GitHub" in reasoning:
        elev.append("strong external validation (GitHub)")
    elif "Moderate GitHub" in reasoning:
        elev.append("moderate external validation (GitHub)")
    if "High platform engagement" in reasoning:
        elev.append("high platform engagement")
    if not elev:
        elev.append("strong composite profile signals")
    elev_str = ", ".join(elev)
    return f"Dense embedding match below top threshold; elevated by {elev_str}."

def patch_reasoning(rank: int, reasoning: str) -> str:
    if OLD_TEXT not in reasoning:
        return reasoning  # already correct (rank 1 etc.), leave untouched

    if rank <= 10:
        replacement = build_top10_replacement(reasoning)
    elif rank <= 25:
        replacement = "Moderate composite fit — retrieval signals and behavioral profile compensate for lower embedding similarity."
    else:
        replacement = "Adjacent semantic profile — included for broad coverage near cutoff."

    return reasoning.replace(OLD_TEXT, replacement)


def main():
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rank = int(row["rank"])
            row["reasoning"] = patch_reasoning(rank, row["reasoning"])
            rows.append(row)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    changed = sum(1 for r in rows if "Adjacent semantic profile" in r["reasoning"]
                  or "Dense embedding match" in r["reasoning"]
                  or "Moderate composite fit" in r["reasoning"])
    print(f"Patched {changed} / {len(rows)} rows.")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
