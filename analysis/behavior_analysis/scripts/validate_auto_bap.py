#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

HUMAN_PATH = Path("analysis/behavior_analysis/data/bap_gold_standard_sample.csv")
AUTO_PATH = Path("analysis/behavior_analysis/auto_annotations/bap_rule_auto_pilot10.csv")


def main() -> None:
    human = pd.read_csv(HUMAN_PATH, keep_default_na=False)
    auto = pd.read_csv(AUTO_PATH, keep_default_na=False)

    merged = human.merge(
        auto,
        on="sample_id",
        how="inner",
        suffixes=("_human", "_auto"),
    )

    print(f"Merged rows: {len(merged)}")
    if len(merged) != len(auto):
        missing = set(auto["sample_id"]) - set(merged["sample_id"])
        print("[warn] sample_ids missing from merge:", sorted(missing))

    # Explicitly use the suffixed columns created by merge
    human_effect_col = "behavioral_effect_human"
    auto_effect_col = "auto_behavioral_effect"

    human_mech_col = "primary_driver_human"
    auto_mech_col = "auto_behavioral_mechanism"

    print("=" * 70)
    print("Behavioral Effect")
    print("=" * 70)

    human_effect = merged[human_effect_col].astype(str).str.strip()
    auto_effect = merged[auto_effect_col].astype(str).str.strip()

    effect_acc = accuracy_score(human_effect, auto_effect)
    print(f"Accuracy: {effect_acc:.3f}\n")
    print(classification_report(human_effect, auto_effect, zero_division=0))

    print("=" * 70)
    print("Behavioral Mechanism")
    print("=" * 70)

    human_mech = merged[human_mech_col].astype(str).str.strip()
    auto_mech = merged[auto_mech_col].astype(str).str.strip()

    mech_acc = accuracy_score(human_mech, auto_mech)
    print(f"Accuracy: {mech_acc:.3f}\n")
    print(classification_report(human_mech, auto_mech, zero_division=0))

    print("=" * 70)
    print("Disagreements")
    print("=" * 70)

    for _, row in merged.iterrows():
        human_eff = str(row[human_effect_col]).strip()
        auto_eff = str(row[auto_effect_col]).strip()
        human_m = str(row[human_mech_col]).strip()
        auto_m = str(row[auto_mech_col]).strip()

        if human_eff != auto_eff or human_m != auto_m:
            print("\nSample:", row["sample_id"])
            print("Human Effect:", human_eff)
            print("Auto Effect :", auto_eff)
            print("Human Mechanism:", human_m)
            print("Auto Mechanism :", auto_m)


if __name__ == "__main__":
    main()