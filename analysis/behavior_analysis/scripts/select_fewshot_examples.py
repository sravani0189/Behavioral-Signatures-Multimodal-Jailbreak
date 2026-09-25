#!/usr/bin/env python3
"""
select_fewshot_examples.py

Greedy Maximum-Coverage Few-shot Selector
=========================================

This script automatically selects a small set of high-quality
few-shot demonstrations from the manually annotated BAP
gold-standard dataset.

Selection objective:

• maximize Behavioral Effect coverage
• maximize Behavioral Mechanism coverage
• maximize Transition coverage
• maximize Model diversity
• maximize Variant diversity
• prefer High-confidence examples
• penalize redundant demonstrations

Outputs

outputs/fewshot/

    fewshot_examples.csv
    fewshot_examples.json
    fewshot_examples.md

    selection_report.md

    selection_trace.csv
    selection_trace.md

    iteration_scores.csv

    coverage_summary.json

Author:
"""

from __future__ import annotations

from pathlib import Path
import json
from math import isnan
from typing import Dict, List, Tuple, Any

import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(
    "/home/spati/projects/MM-SafetyBench/analysis/behavior_analysis"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "bap_gold_standard_sample.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "fewshot"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_EXAMPLES = 12

WEIGHTS = {
    "behavioral_effect": 8,
    "primary_driver": 8,
    "transition": 4,
    "model": 1,
    "variant": 1,
    "confidence": 2,
}

REDUNDANCY_PENALTY = 3

# Optional normalization map for known label issues.
# Keep empty if the dataset is already cleaned.
NORMALIZATION_MAP = {
    # "Visual Groundingt": "Visual Grounding",
    # "MISSING": "MISSING",
}


# ==========================================================
# Utility Functions
# ==========================================================

def normalize_value(value: Any) -> str:
    """
    Convert missing values to a stable string and apply any
    optional label normalization.
    """
    if value is None:
        return "MISSING"

    try:
        if pd.isna(value):
            return "MISSING"
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return "MISSING"

    return NORMALIZATION_MAP.get(text, text)


def confidence_bonus(confidence: Any) -> int:
    if normalize_value(confidence).lower() == "high":
        return WEIGHTS["confidence"]
    return 0


def rarity_bonus(value: Any, frequency_table: Dict[str, int]) -> float:
    """
    Rare categories receive a slightly higher score.

    bonus = 1 / frequency
    """
    key = normalize_value(value)
    freq = frequency_table.get(key, 1)
    return 1.0 / float(freq)


def redundancy_penalty(row: pd.Series, selected_rows: List[pd.Series]) -> int:
    """
    Penalize repeated (effect, driver, transition) combinations.
    """
    row_effect = normalize_value(row.get("behavioral_effect"))
    row_driver = normalize_value(row.get("primary_driver"))
    row_transition = normalize_value(row.get("transition"))

    for s in selected_rows:
        if (
            normalize_value(s.get("behavioral_effect")) == row_effect
            and normalize_value(s.get("primary_driver")) == row_driver
            and normalize_value(s.get("transition")) == row_transition
        ):
            return REDUNDANCY_PENALTY

    return 0


def build_frequency_table(series: pd.Series) -> Dict[str, int]:
    cleaned = series.map(normalize_value)
    return cleaned.value_counts(dropna=False).to_dict()


def compute_gain(
    row: pd.Series,
    covered: Dict[str, set],
    selected_rows: List[pd.Series],
    effect_frequency: Dict[str, int],
    driver_frequency: Dict[str, int],
) -> Tuple[float, Dict[str, str]]:
    """
    Compute marginal gain for a candidate row.
    Returns:
        gain, newly_added
    """
    gain = 0.0

    newly_added = {
        "behavioral_effect": "",
        "primary_driver": "",
        "transition": "",
        "model": "",
        "variant": "",
    }

    row_effect = normalize_value(row.get("behavioral_effect"))
    row_driver = normalize_value(row.get("primary_driver"))
    row_transition = normalize_value(row.get("transition"))
    row_model = normalize_value(row.get("model"))
    row_variant = normalize_value(row.get("variant"))

    if row_effect not in covered["behavioral_effect"]:
        gain += WEIGHTS["behavioral_effect"]
        gain += rarity_bonus(row_effect, effect_frequency)
        newly_added["behavioral_effect"] = row_effect

    if row_driver not in covered["primary_driver"]:
        gain += WEIGHTS["primary_driver"]
        gain += rarity_bonus(row_driver, driver_frequency)
        newly_added["primary_driver"] = row_driver

    if row_transition not in covered["transition"]:
        gain += WEIGHTS["transition"]
        newly_added["transition"] = row_transition

    if row_model not in covered["model"]:
        gain += WEIGHTS["model"]
        newly_added["model"] = row_model

    if row_variant not in covered["variant"]:
        gain += WEIGHTS["variant"]
        newly_added["variant"] = row_variant

    gain += confidence_bonus(row.get("confidence"))
    gain -= redundancy_penalty(row, selected_rows)

    return gain, newly_added


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = {
        "sample_id",
        "model",
        "variant",
        "transition",
        "original_question",
        "changed_question",
        "prompt",
        "original_response",
        "masked_response",
        "evidence",
        "behavioral_effect",
        "primary_driver",
        "confidence",
        "notes",
    }

    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def select_examples(
    df: pd.DataFrame,
    num_examples: int = NUM_EXAMPLES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Greedy maximum-coverage selection.

    Returns:
        selected_df
        selection_trace_df
        iteration_scores_df
        summary dict
    """
    # Prepare frequency tables from the full dataset.
    effect_frequency = build_frequency_table(df["behavioral_effect"])
    driver_frequency = build_frequency_table(df["primary_driver"])

    # Track category coverage.
    covered = {
        "behavioral_effect": set(),
        "primary_driver": set(),
        "transition": set(),
        "model": set(),
        "variant": set(),
    }

    selected_rows: List[pd.Series] = []
    selected_ids = set()

    selection_trace: List[Dict[str, Any]] = []
    all_iteration_scores: List[pd.DataFrame] = []

    for iteration in range(num_examples):
        best_row = None
        best_gain = float("-inf")
        best_newly_added = None
        iter_scores: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            sample_id = normalize_value(row.get("sample_id"))
            if sample_id in selected_ids:
                continue

            gain, newly_added = compute_gain(
                row=row,
                covered=covered,
                selected_rows=selected_rows,
                effect_frequency=effect_frequency,
                driver_frequency=driver_frequency,
            )

            iter_scores.append({
                "iteration": iteration + 1,
                "sample_id": sample_id,
                "gain": gain,
                "behavioral_effect": normalize_value(row.get("behavioral_effect")),
                "primary_driver": normalize_value(row.get("primary_driver")),
                "transition": normalize_value(row.get("transition")),
                "model": normalize_value(row.get("model")),
                "variant": normalize_value(row.get("variant")),
                "confidence": normalize_value(row.get("confidence")),
            })

            # Tie-breakers:
            # 1) higher gain
            # 2) higher confidence bonus
            # 3) smaller sample_id for deterministic output
            if best_row is None:
                best_row = row
                best_gain = gain
                best_newly_added = newly_added
                continue

            best_conf = confidence_bonus(best_row.get("confidence"))
            cur_conf = confidence_bonus(row.get("confidence"))

            if (
                gain > best_gain
                or (
                    gain == best_gain
                    and cur_conf > best_conf
                )
                or (
                    gain == best_gain
                    and cur_conf == best_conf
                    and sample_id < normalize_value(best_row.get("sample_id"))
                )
            ):
                best_row = row
                best_gain = gain
                best_newly_added = newly_added

        if best_row is None:
            raise RuntimeError("No selectable row found. Check input data.")

        selected_rows.append(best_row)
        selected_ids.add(normalize_value(best_row.get("sample_id")))

        covered["behavioral_effect"].add(normalize_value(best_row.get("behavioral_effect")))
        covered["primary_driver"].add(normalize_value(best_row.get("primary_driver")))
        covered["transition"].add(normalize_value(best_row.get("transition")))
        covered["model"].add(normalize_value(best_row.get("model")))
        covered["variant"].add(normalize_value(best_row.get("variant")))

        selection_trace.append({
            "iteration": iteration + 1,
            "sample_id": normalize_value(best_row.get("sample_id")),
            "gain": round(float(best_gain), 4),
            "new_behavioral_effect": best_newly_added["behavioral_effect"],
            "new_primary_driver": best_newly_added["primary_driver"],
            "new_transition": best_newly_added["transition"],
            "new_model": best_newly_added["model"],
            "new_variant": best_newly_added["variant"],
            "confidence": normalize_value(best_row.get("confidence")),
        })

        all_iteration_scores.append(pd.DataFrame(iter_scores))

    selected_df = pd.DataFrame([row.to_dict() for row in selected_rows])
    selection_trace_df = pd.DataFrame(selection_trace)
    iteration_scores_df = pd.concat(all_iteration_scores, ignore_index=True) if all_iteration_scores else pd.DataFrame()

    summary = build_selection_summary(
        df=df,
        selected_df=selected_df,
    )

    return selected_df, selection_trace_df, iteration_scores_df, summary


def build_selection_summary(
    df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build counts and coverage statistics for the report.
    """
    summary: Dict[str, Any] = {}

    dimensions = [
        ("behavioral_effect", "Behavioral Effects"),
        ("primary_driver", "Primary Drivers"),
        ("transition", "Transitions"),
        ("model", "Models"),
        ("variant", "Variants"),
    ]

    for col, label in dimensions:
        full_values = sorted({normalize_value(v) for v in df[col].tolist()})
        selected_values = sorted({normalize_value(v) for v in selected_df[col].tolist()})
        coverage_pct = (len(selected_values) / max(len(full_values), 1)) * 100.0

        summary[col] = {
            "label": label,
            "covered": len(selected_values),
            "total": len(full_values),
            "coverage_pct": coverage_pct,
            "selected_values": selected_values,
            "all_values": full_values,
        }

    return summary


def md_escape(text: Any) -> str:
    """
    Minimal Markdown-safe escaping for pipes and backticks.
    """
    s = normalize_value(text)
    return s.replace("|", "\\|").replace("`", "\\`")


def format_table_markdown(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_No rows._"

    rows = []
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows.append(header)
    rows.append(sep)

    for _, row in df[columns].iterrows():
        cells = [md_escape(row[col]) for col in columns]
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def build_full_example_md(row: pd.Series, selected_reason: Dict[str, Any]) -> str:
    """
    Render one selected example in the full annotation style.
    """
    parts: List[str] = []
    sample_id = normalize_value(row.get("sample_id"))

    parts.append("=" * 60)
    parts.append(f"## Example: {sample_id}")
    parts.append("")
    parts.append("### Selected because")
    parts.append("")
    if selected_reason.get("new_behavioral_effect"):
        parts.append(f"✓ First {md_escape(selected_reason['new_behavioral_effect'])} effect")
    if selected_reason.get("new_primary_driver"):
        parts.append(f"✓ First {md_escape(selected_reason['new_primary_driver'])} driver")
    if selected_reason.get("new_transition"):
        parts.append(f"✓ First {md_escape(selected_reason['new_transition'])} transition")
    if selected_reason.get("new_model"):
        parts.append(f"✓ First {md_escape(selected_reason['new_model'])} model")
    if selected_reason.get("new_variant"):
        parts.append(f"✓ First {md_escape(selected_reason['new_variant'])} variant")
    parts.append(f"✓ Gain = {selected_reason.get('gain', 0)}")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("### Original Question")
    parts.append("")
    parts.append(str(row.get("original_question", "")).strip() or "_None_")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("### Changed Question")
    parts.append("")
    parts.append(str(row.get("changed_question", "")).strip() or "_None_")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("### Prompt")
    parts.append("")
    parts.append(str(row.get("prompt", "")).strip() or "_None_")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("### Original Response")
    parts.append("")
    parts.append(str(row.get("original_response", "")).strip() or "_None_")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("### Masked Response")
    parts.append("")
    parts.append(str(row.get("masked_response", "")).strip() or "_None_")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("### Evidence")
    parts.append("")
    parts.append(str(row.get("evidence", "")).strip() or "_None_")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("### Behavioral Effect")
    parts.append("")
    parts.append(f"**{md_escape(row.get('behavioral_effect'))}**")
    parts.append("")
    parts.append("### Primary Driver")
    parts.append("")
    parts.append(f"**{md_escape(row.get('primary_driver'))}**")
    parts.append("")
    parts.append("### Transition")
    parts.append("")
    parts.append(f"**{md_escape(row.get('transition'))}**")
    parts.append("")
    parts.append("### Model")
    parts.append("")
    parts.append(f"**{md_escape(row.get('model'))}**")
    parts.append("")
    parts.append("### Variant")
    parts.append("")
    parts.append(f"**{md_escape(row.get('variant'))}**")
    parts.append("")
    parts.append("### Confidence")
    parts.append("")
    parts.append(f"**{md_escape(row.get('confidence'))}**")
    parts.append("")
    parts.append("### Notes")
    parts.append("")
    notes = str(row.get("notes", "")).strip()
    parts.append(notes if notes else "_None_")
    parts.append("")

    return "\n".join(parts)


def write_selection_report(
    df: pd.DataFrame,
    selected_df: pd.DataFrame,
    summary: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Human-readable report for paper / appendix.
    """
    lines: List[str] = []
    lines.append("# Few-shot Selection Report")
    lines.append("")
    lines.append(f"Examples selected: **{len(selected_df)}**")
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append("| Dimension | Covered | Total | Coverage |")
    lines.append("| --- | ---: | ---: | ---: |")

    for key in ["behavioral_effect", "primary_driver", "transition", "model", "variant"]:
        item = summary[key]
        lines.append(
            f"| {item['label']} | {item['covered']} | {item['total']} | {item['coverage_pct']:.1f}% |"
        )

    lines.append("")
    lines.append("## Selected Values")
    lines.append("")

    for key in ["behavioral_effect", "primary_driver", "transition", "model", "variant"]:
        item = summary[key]
        lines.append(f"### {item['label']}")
        lines.append("")
        for value in item["selected_values"]:
            lines.append(f"- {value}")
        lines.append("")

    lines.append("## Selection by Iteration")
    lines.append("")
    trace_cols = [
        "iteration",
        "sample_id",
        "gain",
        "new_behavioral_effect",
        "new_primary_driver",
        "new_transition",
        "new_model",
        "new_variant",
        "confidence",
    ]
    trace_preview = selected_df.copy()
    trace_preview = pd.DataFrame()  # placeholder to avoid accidental reuse

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")


def write_fewshot_markdown(
    selected_df: pd.DataFrame,
    selection_trace_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Full demonstration file for GPT / prompt construction.
    Includes the full annotation structure for each selected example.
    """
    trace_lookup = {
        normalize_value(row["sample_id"]): row.to_dict()
        for _, row in selection_trace_df.iterrows()
    }

    lines: List[str] = []
    lines.append("# Few-shot Examples")
    lines.append("")
    lines.append(f"Total examples: **{len(selected_df)}**")
    lines.append("")

    for _, row in selected_df.iterrows():
        sample_id = normalize_value(row.get("sample_id"))
        reason = trace_lookup.get(sample_id, {})
        lines.append(build_full_example_md(row, reason))
        lines.append("\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_fewshot_json(selected_df: pd.DataFrame, output_path: Path) -> None:
    records = selected_df.to_dict(orient="records")
    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_coverage_summary(summary: Dict[str, Any], output_path: Path) -> None:
    serializable = {}
    for key, item in summary.items():
        serializable[key] = {
            "label": item["label"],
            "covered": item["covered"],
            "total": item["total"],
            "coverage_pct": round(item["coverage_pct"], 2),
            "selected_values": item["selected_values"],
            "all_values": item["all_values"],
        }

    output_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 60)
    print("Loading Gold Standard Dataset")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    validate_columns(df)

    # Normalize the working columns used for scoring.
    # Keep the original raw columns intact for exports.
    for col in ["behavioral_effect", "primary_driver", "transition", "model", "variant", "confidence"]:
        df[col] = df[col].map(normalize_value)

    print(f"Loaded {len(df)} samples")
    print("")
    print("Computing greedy selection...")
    print("")

    selected_df, selection_trace_df, iteration_scores_df, summary = select_examples(
        df=df,
        num_examples=NUM_EXAMPLES,
    )

    # Keep the output in the original column order, where possible.
    ordered_columns = [
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
    present_columns = [c for c in ordered_columns if c in selected_df.columns]
    selected_df = selected_df[present_columns].copy()

    # Write outputs.
    selected_df.to_csv(OUTPUT_DIR / "fewshot_examples.csv", index=False)
    write_fewshot_json(selected_df, OUTPUT_DIR / "fewshot_examples.json")
    write_fewshot_markdown(selected_df, selection_trace_df, OUTPUT_DIR / "fewshot_examples.md")

    selection_trace_df.to_csv(OUTPUT_DIR / "selection_trace.csv", index=False)
    selection_trace_df.to_csv(OUTPUT_DIR / "selection_trace.csv", index=False)

    if not iteration_scores_df.empty:
        iteration_scores_df.to_csv(OUTPUT_DIR / "iteration_scores.csv", index=False)

    write_selection_report(
        df=df,
        selected_df=selected_df,
        summary=summary,
        output_path=OUTPUT_DIR / "selection_report.md",
    )

    write_coverage_summary(summary, OUTPUT_DIR / "coverage_summary.json")

    # Optional convenience file for quick inspection.
    selection_trace_df.to_csv(OUTPUT_DIR / "selection_trace.csv", index=False)

    print("Finished.")
    print(f"Results written to\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()