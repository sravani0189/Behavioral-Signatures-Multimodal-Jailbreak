import pandas as pd
from pathlib import Path

DATA = Path(
    "/home/spati/projects/MM-SafetyBench/analysis/behavior_analysis/data/bap_gold_standard_sample.csv"
)

df = pd.read_csv(DATA)

RARE_EFFECTS = [
    "Refusal Removed",
    "Refusal Introduced",
    "Assistance Reduced",
]

RARE_DRIVERS = [
    "OCR Dependency",
    "Operational Reinterpretation",
    "Harmful Intent Ignored",
    "Behavior Shift",
]

print("=" * 80)
print("RARE BEHAVIORAL EFFECTS")
print("=" * 80)

for effect in RARE_EFFECTS:
    subset = df[df["behavioral_effect"] == effect]

    print(f"\n{effect} ({len(subset)})")

    print(
        subset[
            [
                "sample_id",
                "model",
                "variant",
                "transition",
                "confidence",
            ]
        ].to_string(index=False)
    )

print("\n")
print("=" * 80)
print("RARE PRIMARY DRIVERS")
print("=" * 80)

for driver in RARE_DRIVERS:
    subset = df[df["primary_driver"] == driver]

    print(f"\n{driver} ({len(subset)})")

    print(
        subset[
            [
                "sample_id",
                "model",
                "variant",
                "transition",
                "behavioral_effect",
                "confidence",
            ]
        ].to_string(index=False)
    )

print("\n")
print("=" * 80)
print("MISSING PRIMARY DRIVER")
print("=" * 80)

missing = df[df["primary_driver"].isna() | (df["primary_driver"] == "MISSING")]

print(
    missing[
        [
            "sample_id",
            "model",
            "variant",
            "transition",
            "behavioral_effect",
        ]
    ].to_string(index=False)
)