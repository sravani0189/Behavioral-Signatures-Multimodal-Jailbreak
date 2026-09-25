#!/usr/bin/env python3
"""
summarize_UNSAFE_to_SAFE_mechanisms.py

Phase 7: Pattern Analysis
- Reads a coded transition-group CSV
- Extracts only standardized behavioral mechanism labels
- Computes frequency and percentage per mechanism
- Saves a transition-group summary CSV and Markdown table
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# -----------------------------
# Hardcoded paths
# -----------------------------
INPUT_PATH = Path("analysis/transition_groups/UNSAFE_to_SAFE.csv")
OUTPUT_PATH = Path("analysis/patterns/UNSAFE_to_SAFE_mechanism_summary.csv")
MARKDOWN_OUTPUT_PATH = Path("analysis/patterns/UNSAFE_to_SAFE_mechanism_summary.md")

# Column names in your coded CSV
MECHANISM_COLUMN = "behavioral_mechanism"
ID_COLUMN = "question_id"

# Split only on semicolon, pipe, or newlines.
# Do NOT split on commas, because commas can appear inside explanations.
SPLIT_RE = re.compile(r"\s*[;|]\s*|\n+")

# Extract just the mechanism label before an em dash / en dash explanation.
# Examples:
#   "Prompt dependency – The model follows..."
#   "Visual grounding — The model relies..."
LABEL_DESC_RE = re.compile(r"^\s*(?P<label>.+?)\s+[–—]\s+.+$")


def normalize_mechanism_text(text: str) -> list[str]:
    if text is None:
        return []

    raw = str(text).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return []

    parts = [p.strip() for p in SPLIT_RE.split(raw) if p.strip()]

    cleaned: list[str] = []
    seen = set()

    for part in parts:
        m = LABEL_DESC_RE.match(part)
        label = m.group("label").strip() if m else part.strip()

        label = label.strip(" \t\r\n\"'")

        key = label.lower()
        if key not in seen and label:
            seen.add(key)
            cleaned.append(label)

    return cleaned


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    if MECHANISM_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{MECHANISM_COLUMN}' not found in {INPUT_PATH}.\n"
            f"Available columns: {list(df.columns)}"
        )

    total_samples = len(df)
    mechanism_counts: Counter[str] = Counter()
    representative_samples: dict[str, list[str]] = defaultdict(list)

    for row_idx, row in df.iterrows():
        sample_id = row[ID_COLUMN] if ID_COLUMN in df.columns else row_idx
        mechanisms = normalize_mechanism_text(row[MECHANISM_COLUMN])

        for mech in mechanisms:
            mechanism_counts[mech] += 1
            representative_samples[mech].append(str(sample_id))

    records = []
    for mech, freq in mechanism_counts.most_common():
        sample_list = representative_samples[mech]

        unique_sample_list = []
        seen_ids = set()
        for sid in sample_list:
            if sid not in seen_ids:
                seen_ids.add(sid)
                unique_sample_list.append(sid)

        records.append(
            {
                "Transition Group": INPUT_PATH.stem,
                "Behavioral Mechanism": mech,
                "Frequency": freq,
                "Percentage of Samples": round((freq / total_samples) * 100, 2) if total_samples else 0.0,
                "Representative Samples": ", ".join(unique_sample_list),
            }
        )

    summary_df = pd.DataFrame(records).sort_values(
        by=["Frequency", "Behavioral Mechanism"],
        ascending=[False, True],
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False)

    print("\nPhase 7 — Behavioral Mechanism Summary\n")
    print(f"Input file: {INPUT_PATH}")
    print(f"Total samples: {total_samples}\n")
    print(summary_df.to_string(index=False))
    print(f"\nSaved CSV to: {OUTPUT_PATH}")

    lines = []
    lines.append(f"# Behavioral Mechanism Summary: {INPUT_PATH.stem}\n")
    lines.append(f"- Total samples: {total_samples}\n")
    lines.append("\n| Behavioral Mechanism | Frequency | Percentage of Samples | Representative Samples |\n")
    lines.append("|---|---:|---:|---|\n")

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['Behavioral Mechanism']} | {row['Frequency']} | "
            f"{row['Percentage of Samples']} | {row['Representative Samples']} |\n"
        )

    MARKDOWN_OUTPUT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Saved Markdown to: {MARKDOWN_OUTPUT_PATH}")


if __name__ == "__main__":
    main()