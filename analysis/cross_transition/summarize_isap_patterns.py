#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

IN_DIR = Path("analysis/cross_transition")

IN_LONG = IN_DIR / "isap_long.csv"
IN_VARIANT_SUMMARY = IN_DIR / "isap_variant_summary.csv"
IN_SOURCE_SUMMARY = IN_DIR / "isap_source_summary.csv"
IN_MECH_VARIANT = IN_DIR / "isap_mechanism_by_variant.csv"
IN_MECH_TRANSITION = IN_DIR / "isap_mechanism_by_transition.csv"
IN_MECH_PROMPT = IN_DIR / "isap_mechanism_by_prompt.csv"
IN_SOURCE_MECH = IN_DIR / "isap_source_by_mechanism.csv"
IN_DOMINANT = IN_DIR / "isap_dominant_source_candidates.csv"

OUT_PATTERN_SUMMARY = IN_DIR / "isap_pattern_summary.csv"
OUT_VARIANT_INFLUENCE = IN_DIR / "isap_variant_influence_summary.csv"
OUT_SOURCE_INFLUENCE = IN_DIR / "isap_source_influence_summary.csv"
OUT_TOP_MECHANISMS = IN_DIR / "isap_top_mechanisms.csv"
OUT_MD = IN_DIR / "isap_pattern_summary.md"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")


def top_n_table(df: pd.DataFrame, sort_cols: List[str], n: int = 15) -> pd.DataFrame:
    return df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(n)


def main() -> None:
    for p in [
        IN_LONG,
        IN_VARIANT_SUMMARY,
        IN_SOURCE_SUMMARY,
        IN_MECH_VARIANT,
        IN_MECH_TRANSITION,
        IN_MECH_PROMPT,
        IN_SOURCE_MECH,
        IN_DOMINANT,
    ]:
        require(p)

    long_df = pd.read_csv(IN_LONG)
    variant_summary = pd.read_csv(IN_VARIANT_SUMMARY)
    source_summary = pd.read_csv(IN_SOURCE_SUMMARY)
    mech_variant = pd.read_csv(IN_MECH_VARIANT)
    mech_transition = pd.read_csv(IN_MECH_TRANSITION)
    mech_prompt = pd.read_csv(IN_MECH_PROMPT)
    source_mech = pd.read_csv(IN_SOURCE_MECH)
    dominant = pd.read_csv(IN_DOMINANT)

    # Basic totals by mechanism for reporting
    mech_totals = (
        long_df.groupby("Behavioral Mechanism", as_index=False)
        .size()
        .rename(columns={"size": "total_occurrences"})
        .sort_values(["total_occurrences", "Behavioral Mechanism"], ascending=[False, True])
    )
    mech_totals.to_csv(OUT_TOP_MECHANISMS, index=False)

    # Pattern summary: how widely each mechanism appears across variants
    pattern = dominant.copy()

    # Add total occurrences from long_df
    total_occ_map = mech_totals.set_index("Behavioral Mechanism")["total_occurrences"].to_dict()
    pattern["total_occurrences"] = pattern["Behavioral Mechanism"].map(total_occ_map).fillna(0).astype(int)

    # Variant support label
    def support_label(n: int) -> str:
        if n == 3:
            return "shared_across_all_variants"
        if n == 2:
            return "shared_across_two_variants"
        if n == 1:
            return "variant_specific"
        return "unknown"

    pattern["support_class"] = pattern["variant_support_count"].fillna(0).astype(int).map(support_label)

    # A simple influence score: how concentrated the mechanism is in its dominant variant
    def influence_score(row: pd.Series) -> float:
        total = row.get("total_occurrences", 0) or 0
        dom = row.get("dominant_variant_count", 0) or 0
        return float(dom) / float(total) if total else 0.0

    pattern["dominance_ratio"] = pattern.apply(influence_score, axis=1)

    # Clean ordering
    pattern = pattern.sort_values(
        ["variant_support_count", "dominant_variant_count", "total_occurrences", "Behavioral Mechanism"],
        ascending=[False, False, False, True],
    )

    # Variant-level influence summary
    variant_influence_rows = []
    for variant in ["SD", "SD_TYPO", "TYPO"]:
        sub = long_df[long_df["variant"] == variant].copy()
        if sub.empty:
            continue

        variant_influence_rows.append(
            {
                "variant": variant,
                "sample_count": int(len(sub)),
                "distinct_mechanisms": int(sub["Behavioral Mechanism"].nunique()),
                "top_mechanism": sub["Behavioral Mechanism"].value_counts().index[0],
                "top_mechanism_count": int(sub["Behavioral Mechanism"].value_counts().iloc[0]),
                "top_source_candidate": (
                    pattern[pattern["dominant_variant"] == variant]["source_candidate"].value_counts().index[0]
                    if not pattern[pattern["dominant_variant"] == variant].empty
                    else ""
                ),
            }
        )

    variant_influence = pd.DataFrame(variant_influence_rows)

    # Source-level influence summary
    source_influence_rows = []
    for source in ["Visual scene", "Embedded OCR text", "Prompt wording", "Multisource / robust across variants", "Visual scene likely influential", "Embedded OCR text likely influential", "Prompt wording or shared instruction likely influential"]:
        sub = pattern[pattern["source_candidate"] == source]
        if sub.empty:
            continue
        source_influence_rows.append(
            {
                "source_candidate": source,
                "mechanism_count": int(len(sub)),
                "total_occurrences": int(sub["total_occurrences"].sum()),
                "shared_across_all_variants": int((sub["support_class"] == "shared_across_all_variants").sum()),
                "variant_specific": int((sub["support_class"] == "variant_specific").sum()),
                "mean_dominance_ratio": round(float(sub["dominance_ratio"].mean()), 4),
            }
        )

    source_influence = pd.DataFrame(source_influence_rows).sort_values(
        ["total_occurrences", "mechanism_count", "source_candidate"],
        ascending=[False, False, True],
    )

    # Save CSVs
    pattern.to_csv(OUT_PATTERN_SUMMARY, index=False)
    variant_influence.to_csv(OUT_VARIANT_INFLUENCE, index=False)
    source_influence.to_csv(OUT_SOURCE_INFLUENCE, index=False)

    # Markdown summary
    md: List[str] = []
    md.append("# Information Source Analysis Summary\n\n")

    md.append("## Variant overview\n\n")
    md.append(variant_summary.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Source overview\n\n")
    md.append(source_summary.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Variant influence summary\n\n")
    if variant_influence.empty:
        md.append("No rows available.\n\n")
    else:
        md.append(variant_influence.to_markdown(index=False))
        md.append("\n\n")

    md.append("## Source influence summary\n\n")
    if source_influence.empty:
        md.append("No rows available.\n\n")
    else:
        md.append(source_influence.to_markdown(index=False))
        md.append("\n\n")

    md.append("## Top mechanisms by total occurrences\n\n")
    md.append(mech_totals.head(20).to_markdown(index=False))
    md.append("\n\n")

    md.append("## Shared vs variant-specific mechanisms\n\n")
    md.append(
        pattern[
            [
                "Behavioral Mechanism",
                "variants_present",
                "variant_support_count",
                "support_class",
                "dominant_variant",
                "dominant_variant_count",
                "source_candidate",
                "dominance_ratio",
            ]
        ]
        .head(30)
        .to_markdown(index=False)
    )
    md.append("\n\n")

    md.append("## Mechanism by variant\n\n")
    md.append(mech_variant.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Mechanism by transition group\n\n")
    md.append(mech_transition.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Mechanism by prompt variant\n\n")
    md.append(mech_prompt.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Source signature by mechanism\n\n")
    md.append(source_mech.to_markdown(index=False))
    md.append("\n")

    OUT_MD.write_text("".join(md), encoding="utf-8")

    print(f"Saved pattern summary -> {OUT_PATTERN_SUMMARY}")
    print(f"Saved variant influence summary -> {OUT_VARIANT_INFLUENCE}")
    print(f"Saved source influence summary -> {OUT_SOURCE_INFLUENCE}")
    print(f"Saved top mechanisms -> {OUT_TOP_MECHANISMS}")
    print(f"Saved markdown report -> {OUT_MD}")


if __name__ == "__main__":
    main()