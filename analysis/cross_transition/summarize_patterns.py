#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

IN_PROFILE = Path("analysis/cross_transition/cross_transition_mechanism_profile.csv")
IN_GROUP_SUMMARY = Path("analysis/cross_transition/cross_transition_group_summary.csv")

OUT_DIR = Path("analysis/cross_transition")
OUT_PATTERNS = OUT_DIR / "cross_transition_pattern_summary.csv"
OUT_MD = OUT_DIR / "cross_transition_pattern_summary.md"

GROUP_ORDER = [
    "SAFE_to_SAFE",
    "SAFE_to_UNSAFE",
    "UNSAFE_to_SAFE",
    "UNSAFE_to_UNSAFE",
]


def main() -> None:
    if not IN_PROFILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_PROFILE}")
    if not IN_GROUP_SUMMARY.exists():
        raise FileNotFoundError(f"Missing input file: {IN_GROUP_SUMMARY}")

    profile = pd.read_csv(IN_PROFILE)
    group_summary = pd.read_csv(IN_GROUP_SUMMARY)

    required = ["Behavioral Mechanism"] + GROUP_ORDER + ["present_groups", "total_frequency", "pattern_family", "stable_total", "change_total"]
    missing = [c for c in required if c not in profile.columns]
    if missing:
        raise KeyError(f"Profile file missing columns: {missing}")

    profile = profile.copy()

    # Additional synthesis labels
    def transition_label(row: pd.Series) -> str:
        if row["present_groups"] == 4:
            return "shared_across_all_groups"
        if row["present_groups"] == 1:
            return "unique_to_one_group"
        if row["stable_total"] > row["change_total"]:
            return "stability_dominant"
        if row["change_total"] > row["stable_total"]:
            return "transition_dominant"
        return "balanced"

    def dominant_side(row: pd.Series) -> str:
        stable = row["stable_total"]
        change = row["change_total"]
        if stable > change:
            return "stable"
        if change > stable:
            return "change"
        return "tie"

    profile["transition_class"] = profile.apply(transition_label, axis=1)
    profile["dominant_side"] = profile.apply(dominant_side, axis=1)

    # A simple numerical balance score: positive = stable-leaning, negative = change-leaning
    total = profile["total_frequency"].replace(0, pd.NA)
    profile["balance_score"] = ((profile["stable_total"] - profile["change_total"]) / total).fillna(0)

    profile = profile.sort_values(
        ["transition_class", "dominant_side", "total_frequency", "Behavioral Mechanism"],
        ascending=[True, True, False, True],
    )

    profile.to_csv(OUT_PATTERNS, index=False)

    # Build markdown summary
    md: List[str] = []
    md.append("# Cross-Transition Pattern Summary\n\n")

    md.append("## Group overview\n\n")
    md.append(group_summary.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Pattern family counts\n\n")
    family_counts = profile.groupby("pattern_family", as_index=False).size().sort_values("size", ascending=False)
    md.append(family_counts.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Transition class counts\n\n")
    class_counts = profile.groupby("transition_class", as_index=False).size().sort_values("size", ascending=False)
    md.append(class_counts.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Top mechanisms by total frequency\n\n")
    top_total = profile[["Behavioral Mechanism", "total_frequency", "transition_class", "dominant_side"]].head(15)
    md.append(top_total.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Stability-leaning mechanisms\n\n")
    stable = profile[profile["transition_class"] == "stability_dominant"]
    if stable.empty:
        md.append("None.\n\n")
    else:
        md.append(stable[["Behavioral Mechanism", "SAFE_to_SAFE", "UNSAFE_to_UNSAFE", "SAFE_to_UNSAFE", "UNSAFE_to_SAFE", "total_frequency"]].head(15).to_markdown(index=False))
        md.append("\n\n")

    md.append("## Transition-leaning mechanisms\n\n")
    change = profile[profile["transition_class"] == "transition_dominant"]
    if change.empty:
        md.append("None.\n\n")
    else:
        md.append(change[["Behavioral Mechanism", "SAFE_to_SAFE", "UNSAFE_to_UNSAFE", "SAFE_to_UNSAFE", "UNSAFE_to_SAFE", "total_frequency"]].head(15).to_markdown(index=False))
        md.append("\n\n")

    md.append("## Mechanisms shared across all four groups\n\n")
    shared = profile[profile["transition_class"] == "shared_across_all_groups"]
    if shared.empty:
        md.append("None.\n\n")
    else:
        md.append(shared[["Behavioral Mechanism", "total_frequency", "balance_score"]].head(20).to_markdown(index=False))
        md.append("\n\n")

    OUT_MD.write_text("".join(md), encoding="utf-8")

    print(f"Saved pattern summary -> {OUT_PATTERNS}")
    print(f"Saved markdown report -> {OUT_MD}")


if __name__ == "__main__":
    main()