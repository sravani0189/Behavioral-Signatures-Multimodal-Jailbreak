#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


TRANSITION_MAP = {
    "SAFE -> SAFE": "SAFE_TO_SAFE",
    "SAFE -> UNSAFE": "SAFE_TO_UNSAFE",
    "UNSAFE -> SAFE": "UNSAFE_TO_SAFE",
    "UNSAFE -> UNSAFE": "UNSAFE_TO_UNSAFE",
}


VALID_GROUPS = {
    "SAFE_TO_SAFE",
    "SAFE_TO_UNSAFE",
    "UNSAFE_TO_SAFE",
    "UNSAFE_TO_UNSAFE",
}


def normalize_transition(value: str) -> str | None:
    text = str(value).strip().upper()
    if text in TRANSITION_MAP:
        return TRANSITION_MAP[text]
    return None


def stable_cell_seed(seed: int, model: str, variant: str, transition_group: str) -> int:
    payload = f"{seed}|{model}|{variant}|{transition_group}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analysis/behavior_analysis/data/behavior_pairs_raw.csv"),
        help="Path to behavior_pairs_raw.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/behavior_analysis/data/bap_gold_standard_sample.csv"),
        help="Output CSV for manual BAP annotation",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("analysis/behavior_analysis/data/bap_gold_sampling_report.csv"),
        help="Output CSV report of available vs sampled counts",
    )
    parser.add_argument(
        "--n-per-cell",
        type=int,
        default=4,
        help="Number of samples to take per model × variant × transition cell.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input file: {args.input}")

    df = pd.read_csv(args.input)

    required_cols = ["sample_id", "model", "variant", "transition"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    df["transition_group"] = df["transition"].apply(normalize_transition)
    before = len(df)

    # Keep only behavioral transitions, exclude API/system blocks and anything invalid.
    df = df[df["transition_group"].isin(VALID_GROUPS)].copy()
    after = len(df)

    sampled_parts = []
    report_rows = []

    for (model, variant, transition_group), group_df in df.groupby(
        ["model", "variant", "transition_group"], dropna=False
    ):
        available = len(group_df)
        if available == 0:
            continue

        n_take = min(args.n_per_cell, available)
        cell_seed = stable_cell_seed(args.seed, model, variant, transition_group)

        sampled = group_df.sample(n=n_take, random_state=cell_seed)
        sampled_parts.append(sampled)

        report_rows.append(
            {
                "model": model,
                "variant": variant,
                "transition_group": transition_group,
                "available": available,
                "sampled": n_take,
            }
        )

    if not sampled_parts:
        raise RuntimeError("No samples were selected. Check the input CSV.")

    out = pd.concat(sampled_parts, ignore_index=True)

    # Sort for readability.
    out = out.sort_values(
        by=["model", "variant", "transition_group", "question_id", "sample_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Add blank annotation fields for manual work.
    for col in ["evidence", "behavioral_effect", "primary_driver", "confidence", "notes"]:
        if col not in out.columns:
            out[col] = ""

    preferred_order = [
        "sample_id",
        "model",
        "question_id",
        "scenario",
        "variant",
        "transition",
        "transition_group",
        "original_question",
        "changed_question",
        "key_phrase",
        "phrase_type",
        "prompt_source",
        "prompt",
        "original_response",
        "masked_response",
        "original_label",
        "masked_label",
        "evidence",
        "behavioral_effect",
        "primary_driver",
        "confidence",
        "notes",
    ]

    final_cols = [c for c in preferred_order if c in out.columns]
    remaining_cols = [c for c in out.columns if c not in final_cols]
    out = out[final_cols + remaining_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    report = pd.DataFrame(report_rows)
    if not report.empty:
        report = report.sort_values(
            by=["model", "variant", "transition_group"],
            kind="stable",
        ).reset_index(drop=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)

    print("=" * 72)
    print(f"Original rows            : {before}")
    print(f"Behavior rows kept       : {after}")
    print(f"API/system blocks removed: {before - after}")
    print("=" * 72)
    print(f"[OK] Wrote {args.output}")
    print(f"[OK] Rows: {len(out)}")
    print(f"[OK] Wrote {args.report}")
    print("[OK] Columns added for manual annotation: evidence, behavioral_effect, primary_driver, confidence, notes")


if __name__ == "__main__":
    main()