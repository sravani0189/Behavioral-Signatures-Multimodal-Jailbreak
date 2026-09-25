#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

INPUT_FILES = [
    "analysis/patterns/SAFE_to_SAFE_mechanism_summary.csv",
    "analysis/patterns/SAFE_to_UNSAFE_mechanism_summary.csv",
    "analysis/patterns/UNSAFE_to_SAFE_mechanism_summary.csv",
    "analysis/patterns/UNSAFE_to_UNSAFE_mechanism_summary.csv",
]

OUT_LONG = Path("analysis/patterns/all_transition_mechanisms_summary.csv")
OUT_PIVOT = Path("analysis/patterns/all_transition_mechanisms_pivot.csv")
OUT_MD = Path("analysis/patterns/all_transition_mechanisms_summary.md")


def main() -> None:
    frames = []

    for f in INPUT_FILES:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)

        # Standardize transition group name from filename if needed
        if "Transition Group" in df.columns:
            group_name = str(df["Transition Group"].iloc[0])
        else:
            group_name = path.stem.replace("_mechanism_summary", "")

        df = df.copy()
        df["transition_group"] = group_name
        frames.append(df)

    long_df = pd.concat(frames, ignore_index=True)

    # Reorder for readability
    cols = [
        "transition_group",
        "Behavioral Mechanism",
        "Frequency",
        "Percentage of Samples",
        "Representative Samples",
    ]
    long_df = long_df[[c for c in cols if c in long_df.columns]]

    long_df.to_csv(OUT_LONG, index=False)

    # Wide pivot: mechanisms vs transition groups
    pivot_df = long_df.pivot_table(
        index="Behavioral Mechanism",
        columns="transition_group",
        values="Frequency",
        aggfunc="first",
        fill_value=0,
    ).reset_index()

    pivot_df.to_csv(OUT_PIVOT, index=False)

    # Markdown version of the long table
    md = []
    md.append("# All Transition Mechanism Summary\n\n")
    md.append(long_df.to_markdown(index=False))
    md.append("\n")

    OUT_MD.write_text("".join(md), encoding="utf-8")

    print(f"Saved long table -> {OUT_LONG}")
    print(f"Saved pivot table -> {OUT_PIVOT}")
    print(f"Saved markdown -> {OUT_MD}")


if __name__ == "__main__":
    main()