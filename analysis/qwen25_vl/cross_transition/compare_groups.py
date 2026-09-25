#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List
import re

import pandas as pd

BASE_DIR = Path("analysis/patterns")
INPUT_FILE = BASE_DIR / "all_transition_mechanisms_summary.csv"

OUT_DIR = Path("analysis/cross_transition")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PIVOT = OUT_DIR / "cross_transition_pivot.csv"
OUT_GROUP_SUMMARY = OUT_DIR / "cross_transition_group_summary.csv"
OUT_SHARED = OUT_DIR / "shared_mechanisms.csv"
OUT_UNIQUE = OUT_DIR / "unique_mechanisms.csv"
OUT_PROFILE = OUT_DIR / "cross_transition_mechanism_profile.csv"
OUT_MD = OUT_DIR / "cross_transition_comparison.md"

GROUP_ORDER = [
    "SAFE_to_SAFE",
    "SAFE_to_UNSAFE",
    "UNSAFE_to_SAFE",
    "UNSAFE_to_UNSAFE",
]

GROUP_ALIASES = {
    "SAFE→SAFE": "SAFE_to_SAFE",
    "SAFE->SAFE": "SAFE_to_SAFE",
    "S→S": "SAFE_to_SAFE",
    "S->S": "SAFE_to_SAFE",
    "SAFE_to_SAFE": "SAFE_to_SAFE",
    "SAFE→UNSAFE": "SAFE_to_UNSAFE",
    "SAFE->UNSAFE": "SAFE_to_UNSAFE",
    "S→U": "SAFE_to_UNSAFE",
    "S->U": "SAFE_to_UNSAFE",
    "SAFE_to_UNSAFE": "SAFE_to_UNSAFE",
    "UNSAFE→SAFE": "UNSAFE_to_SAFE",
    "UNSAFE->SAFE": "UNSAFE_to_SAFE",
    "U→S": "UNSAFE_to_SAFE",
    "U->S": "UNSAFE_to_SAFE",
    "UNSAFE_to_SAFE": "UNSAFE_to_SAFE",
    "UNSAFE→UNSAFE": "UNSAFE_to_UNSAFE",
    "UNSAFE->UNSAFE": "UNSAFE_to_UNSAFE",
    "U→U": "UNSAFE_to_UNSAFE",
    "U->U": "UNSAFE_to_UNSAFE",
    "UNSAFE_to_UNSAFE": "UNSAFE_to_UNSAFE",
}


def normalize_group(value: object) -> str:
    text = str(value).strip()
    return GROUP_ALIASES.get(text, text)


def normalize_mechanism(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[.;:,]+$", "", text)  # remove trailing punctuation
    text = re.sub(r"\s+", " ", text)      # collapse extra spaces
    return text


def get_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find a {label} column. Tried: {candidates}. Found: {list(df.columns)}"
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing combined summary file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    group_col = get_column(df, ["transition_group", "Transition Group"], "transition group")
    mech_col = get_column(df, ["Behavioral Mechanism", "behavioral_mechanism", "mechanism"], "mechanism")
    freq_col = get_column(df, ["Frequency", "frequency"], "frequency")

    working = df[[group_col, mech_col, freq_col]].copy()
    working.columns = ["transition_group", "Behavioral Mechanism", "Frequency"]

    working["transition_group"] = working["transition_group"].map(normalize_group)
    working["Behavioral Mechanism"] = working["Behavioral Mechanism"].apply(normalize_mechanism)
    working["Frequency"] = pd.to_numeric(working["Frequency"], errors="coerce").fillna(0).astype(int)

    working = working[working["transition_group"].isin(GROUP_ORDER)].copy()

    if working.empty:
        raise ValueError("No rows matched the expected transition groups.")

    # Long summary preserved for downstream analysis
    working.to_csv(OUT_DIR / "cross_transition_long.csv", index=False)

    # Pivot table: mechanisms x groups
    pivot = (
        working.pivot_table(
            index="Behavioral Mechanism",
            columns="transition_group",
            values="Frequency",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=GROUP_ORDER, fill_value=0)
        .reset_index()
    )

    # Add cross-group descriptors
    group_cols = GROUP_ORDER
    pivot["present_groups"] = (pivot[group_cols] > 0).sum(axis=1)
    pivot["total_frequency"] = pivot[group_cols].sum(axis=1)
    pivot["dominant_group"] = pivot[group_cols].idxmax(axis=1)
    pivot["dominant_frequency"] = pivot[group_cols].max(axis=1)
    pivot["stable_total"] = pivot["SAFE_to_SAFE"] + pivot["UNSAFE_to_UNSAFE"]
    pivot["change_total"] = pivot["SAFE_to_UNSAFE"] + pivot["UNSAFE_to_SAFE"]

    def classify(row: pd.Series) -> str:
        if row["present_groups"] == 4:
            return "shared_across_all_groups"
        if row["present_groups"] == 1:
            return "unique_to_one_group"
        if row["stable_total"] > row["change_total"]:
            return "stable_leaning"
        if row["change_total"] > row["stable_total"]:
            return "change_leaning"
        return "balanced"

    pivot["pattern_family"] = pivot.apply(classify, axis=1)

    pivot.to_csv(OUT_PIVOT, index=False)
    pivot.to_csv(OUT_PROFILE, index=False)

    # Shared mechanisms
    shared = pivot[pivot["present_groups"] == 4].copy()
    shared = shared.sort_values(["total_frequency", "Behavioral Mechanism"], ascending=[False, True])
    shared.to_csv(OUT_SHARED, index=False)

    # Unique mechanisms
    unique = pivot[pivot["present_groups"] == 1].copy()
    unique["unique_group"] = unique["dominant_group"]
    unique = unique.sort_values(
        ["unique_group", "total_frequency", "Behavioral Mechanism"],
        ascending=[True, False, True],
    )
    unique.to_csv(OUT_UNIQUE, index=False)

    # Group-level summary
    group_rows = []
    for group in GROUP_ORDER:
        sub = working[working["transition_group"] == group].copy()
        mech_counts = (
            sub.groupby("Behavioral Mechanism", as_index=False)["Frequency"]
            .sum()
            .sort_values(["Frequency", "Behavioral Mechanism"], ascending=[False, True])
        )

        total_freq = int(sub["Frequency"].sum())
        unique_mechs = int((pivot["present_groups"] == 1).sum())
        dominant_mech = mech_counts.iloc[0]["Behavioral Mechanism"] if not mech_counts.empty else ""
        dominant_freq = int(mech_counts.iloc[0]["Frequency"]) if not mech_counts.empty else 0

        group_rows.append(
            {
                "transition_group": group,
                "total_frequency": total_freq,
                "distinct_mechanisms": int(sub["Behavioral Mechanism"].nunique()),
                "dominant_mechanism": dominant_mech,
                "dominant_mechanism_frequency": dominant_freq,
                "shared_mechanisms_across_all_groups": int((pivot["present_groups"] == 4).sum()),
                "unique_mechanisms_in_dataset": unique_mechs,
            }
        )

    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(OUT_GROUP_SUMMARY, index=False)

    # Markdown report
    md: List[str] = []
    md.append("# Cross-Transition Comparison Summary\n\n")

    md.append("## Group-level overview\n\n")
    md.append(group_summary.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Shared mechanisms across all four transition groups\n\n")
    if shared.empty:
        md.append("None.\n\n")
    else:
        md.append(
            shared[
                [
                    "Behavioral Mechanism",
                    "SAFE_to_SAFE",
                    "SAFE_to_UNSAFE",
                    "UNSAFE_to_SAFE",
                    "UNSAFE_to_UNSAFE",
                    "total_frequency",
                ]
            ].to_markdown(index=False)
        )
        md.append("\n\n")

    md.append("## Unique mechanisms appearing in only one group\n\n")
    if unique.empty:
        md.append("None.\n\n")
    else:
        md.append(unique[["Behavioral Mechanism", "unique_group", "total_frequency"]].to_markdown(index=False))
        md.append("\n\n")

    md.append("## Mechanism profile table\n\n")
    md.append(
        pivot[
            [
                "Behavioral Mechanism",
                "SAFE_to_SAFE",
                "SAFE_to_UNSAFE",
                "UNSAFE_to_SAFE",
                "UNSAFE_to_UNSAFE",
                "present_groups",
                "total_frequency",
                "pattern_family",
            ]
        ].to_markdown(index=False)
    )
    md.append("\n")

    OUT_MD.write_text("".join(md), encoding="utf-8")

    print(f"Saved pivot table -> {OUT_PIVOT}")
    print(f"Saved group summary -> {OUT_GROUP_SUMMARY}")
    print(f"Saved shared mechanisms -> {OUT_SHARED}")
    print(f"Saved unique mechanisms -> {OUT_UNIQUE}")
    print(f"Saved mechanism profile -> {OUT_PROFILE}")
    print(f"Saved markdown report -> {OUT_MD}")


if __name__ == "__main__":
    main()