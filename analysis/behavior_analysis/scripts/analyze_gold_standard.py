#!/usr/bin/env python3
"""
analyze_gold_standard.py

Analyze the BAP Gold Standard annotations and generate
summary statistics for few-shot selection.

Author: Your Name
"""

from pathlib import Path
import json
import pandas as pd


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path("/home/spati/projects/MM-SafetyBench/analysis/behavior_analysis")

DATA_PATH = PROJECT_ROOT / "data" / "bap_gold_standard_sample.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Helper Functions
# ==========================================================

def analyze_column(df: pd.DataFrame, column: str):
    """
    Generate frequency table for a categorical column.
    """

    table = (
        df[column]
        .fillna("MISSING")
        .value_counts()
        .rename_axis(column)
        .reset_index(name="count")
    )

    table["percent"] = (
        table["count"] / len(df) * 100
    ).round(2)

    return table


def save_distribution(df, column):
    """
    Save distribution CSV.
    """

    table = analyze_column(df, column)

    filename = OUTPUT_DIR / f"{column}_distribution.csv"

    table.to_csv(filename, index=False)

    return table


def save_crosstab(df, row, col):
    """
    Save cross-tabulation.
    """

    table = pd.crosstab(df[row], df[col])

    filename = OUTPUT_DIR / f"{row}_vs_{col}.csv"

    table.to_csv(filename)

    return table


# ==========================================================
# Markdown Report
# ==========================================================

def generate_report(distributions, summary):

    report = []

    report.append("# Gold Standard Dataset Analysis\n")

    report.append("## Dataset Summary\n")

    report.append(f"- Total Samples: **{summary['num_samples']}**")
    report.append(f"- Unique Models: **{summary['num_models']}**")
    report.append(f"- Unique Variants: **{summary['num_variants']}**")
    report.append(f"- Unique Transitions: **{summary['num_transitions']}**")
    report.append(f"- Behavioral Effects: **{summary['num_behavioral_effects']}**")
    report.append(f"- Behavioral Mechanisms: **{summary['num_primary_drivers']}**")

    report.append("\n---\n")

    for column, table in distributions.items():

        report.append(f"## {column.replace('_', ' ').title()}\n")

        report.append(table.to_markdown(index=False))

        report.append("\n")

    report_path = OUTPUT_DIR / "analysis_report.md"

    with open(report_path, "w") as f:
        f.write("\n".join(report))


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Loading Gold Standard Dataset")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df)} samples\n")

    expected_columns = [
        "behavioral_effect",
        "primary_driver",
        "transition",
        "model",
        "variant",
        "confidence",
    ]

    print("Checking required columns...")

    for col in expected_columns:

        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print("✓ All required columns found\n")

    distributions = {}

    print("=" * 60)
    print("Generating distributions")
    print("=" * 60)

    for column in expected_columns:

        print(f"Analyzing {column}")

        table = save_distribution(df, column)

        distributions[column] = table

    print()

    print("=" * 60)
    print("Generating cross-tabulations")
    print("=" * 60)

    save_crosstab(df, "behavioral_effect", "primary_driver")
    save_crosstab(df, "behavioral_effect", "transition")
    save_crosstab(df, "model", "behavioral_effect")
    save_crosstab(df, "variant", "behavioral_effect")
    save_crosstab(df, "transition", "model")

    summary = {

        "num_samples": len(df),

        "num_models": df["model"].nunique(),

        "num_variants": df["variant"].nunique(),

        "num_transitions": df["transition"].nunique(),

        "num_behavioral_effects": df["behavioral_effect"].nunique(),

        "num_primary_drivers": df["primary_driver"].nunique(),
    }

    with open(OUTPUT_DIR / "coverage_summary.json", "w") as f:

        json.dump(summary, f, indent=4)

    generate_report(distributions, summary)

    print()
    print("=" * 60)
    print("Analysis Complete")
    print("=" * 60)

    print(f"Output directory:\n{OUTPUT_DIR}")

    print("\nGenerated files:")

    for file in sorted(OUTPUT_DIR.glob("*")):

        print(" -", file.name)


if __name__ == "__main__":
    main()