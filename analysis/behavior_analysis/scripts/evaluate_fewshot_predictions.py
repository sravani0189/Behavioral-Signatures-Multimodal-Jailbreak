#!/usr/bin/env python3
"""
evaluate_fewshot_predictions.py

Evaluate few-shot GPT predictions against gold labels.

This script:
  1. Loads the gold CSV and prediction CSV
  2. Matches rows by sample_id
  3. Computes metrics for:
       - Behavioral Effect
       - Behavioral Mechanism (Primary Driver)
       - Transition
       - Confidence
  4. Writes:
       - metrics.md
       - merged_predictions.csv
       - per-task classification reports
       - per-task confusion matrices

Example:
  python analysis/behavior_analysis/scripts/evaluate_fewshot_predictions.py \
    --gold-csv analysis/behavior_analysis/data/bap_gold_standard_sample.csv \
    --pred-csv analysis/behavior_analysis/outputs/fewshot/predictions_50.csv \
    --output-dir analysis/behavior_analysis/outputs/fewshot/eval_50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path("/home/spati/projects/MM-SafetyBench/analysis/behavior_analysis")
DEFAULT_GOLD_CSV = PROJECT_ROOT / "data" / "bap_gold_standard_sample.csv"
DEFAULT_PRED_CSV = PROJECT_ROOT / "outputs" / "fewshot" / "predictions_50.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fewshot" / "eval_50"

TASKS: List[Tuple[str, str, str]] = [
    ("Behavioral Effect", "behavioral_effect_gold", "behavioral_effect_pred"),
    ("Behavioral Mechanism", "primary_driver_gold", "primary_driver_pred"),
    ("Transition", "transition_gold", "transition_pred"),
    ("Confidence", "confidence_gold", "confidence_pred"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate few-shot GPT predictions.")
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD_CSV)
    parser.add_argument("--pred-csv", type=Path, default=DEFAULT_PRED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def ensure_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def write_markdown_report(
    output_dir: Path,
    merged: pd.DataFrame,
    results: List[Dict[str, object]],
) -> None:
    lines: List[str] = []
    lines.append("# Few-shot Evaluation Report")
    lines.append("")
    lines.append(f"Matched samples: **{len(merged)}**")
    lines.append("")

    for result in results:
        lines.append(f"## {result['task_name']}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        lines.append(f"| Accuracy | {result['accuracy']:.3f} |")
        lines.append(f"| Macro Precision | {result['macro_precision']:.3f} |")
        lines.append(f"| Macro Recall | {result['macro_recall']:.3f} |")
        lines.append(f"| Macro F1 | {result['macro_f1']:.3f} |")
        lines.append(f"| Weighted F1 | {result['weighted_f1']:.3f} |")
        lines.append("")
        lines.append("Per-class report and confusion matrix are saved as CSV files in this directory.")
        lines.append("")

    lines.append("## Prediction Coverage")
    lines.append("")
    for _, _, pred_col in TASKS:
        if pred_col in merged.columns:
            filled = (normalize_text(merged[pred_col]) != "").sum()
            lines.append(f"- {pred_col}: {filled}/{len(merged)} non-empty")
    lines.append("")

    (output_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_task(
    merged: pd.DataFrame,
    task_name: str,
    gold_col: str,
    pred_col: str,
    output_dir: Path,
) -> Dict[str, object]:
    gold = normalize_text(merged[gold_col])
    pred = normalize_text(merged[pred_col])

    labels = sorted((set(gold.tolist()) | set(pred.tolist())) - {""})

    accuracy = accuracy_score(gold, pred)

    if labels:
        report_dict = classification_report(
            gold,
            pred,
            labels=labels,
            zero_division=0,
            output_dict=True,
        )
        cm = confusion_matrix(gold, pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    else:
        report_dict = {
            "macro avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0},
            "weighted avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0},
        }
        cm_df = pd.DataFrame()

    report_df = pd.DataFrame(report_dict).transpose()

    safe_name = task_name.lower().replace(" ", "_")
    report_df.to_csv(output_dir / f"{safe_name}_report.csv")
    cm_df.to_csv(output_dir / f"{safe_name}_confusion.csv")

    return {
        "task_name": task_name,
        "accuracy": accuracy,
        "macro_precision": float(report_dict.get("macro avg", {}).get("precision", 0.0)),
        "macro_recall": float(report_dict.get("macro avg", {}).get("recall", 0.0)),
        "macro_f1": float(report_dict.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)),
        "report_df": report_df,
        "cm_df": cm_df,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.gold_csv.exists():
        raise FileNotFoundError(f"Gold CSV not found: {args.gold_csv}")
    if not args.pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {args.pred_csv}")

    gold = pd.read_csv(args.gold_csv)
    pred = pd.read_csv(args.pred_csv)

    ensure_columns(gold, ["sample_id"], "Gold CSV")
    ensure_columns(pred, ["sample_id"], "Prediction CSV")

    gold_required = ["behavioral_effect", "primary_driver", "transition", "confidence"]
    pred_required = [
        "behavioral_effect_pred",
        "primary_driver_pred",
        "transition_pred",
        "confidence_pred",
    ]

    ensure_columns(gold, gold_required, "Gold CSV")
    ensure_columns(pred, pred_required, "Prediction CSV")

    gold_sub = gold[
        [
            "sample_id",
            "behavioral_effect",
            "primary_driver",
            "transition",
            "confidence",
        ]
    ].copy()

    pred_sub = pred[
        [
            "sample_id",
            "behavioral_effect_pred",
            "primary_driver_pred",
            "transition_pred",
            "confidence_pred",
        ]
    ].copy()

    gold_sub = gold_sub.rename(
        columns={
            "behavioral_effect": "behavioral_effect_gold",
            "primary_driver": "primary_driver_gold",
            "transition": "transition_gold",
            "confidence": "confidence_gold",
        }
    )

    merged = gold_sub.merge(pred_sub, on="sample_id", how="inner")

    if merged.empty:
        raise ValueError("No overlapping sample_id values found between gold and prediction CSVs.")

    merged_out_cols = [
        "sample_id",
        "behavioral_effect_gold",
        "primary_driver_gold",
        "transition_gold",
        "confidence_gold",
        "behavioral_effect_pred",
        "primary_driver_pred",
        "transition_pred",
        "confidence_pred",
    ]
    merged[merged_out_cols].to_csv(args.output_dir / "merged_predictions.csv", index=False)

    results: List[Dict[str, object]] = []
    for task_name, gold_col, pred_col in TASKS:
        results.append(
            evaluate_task(
                merged=merged,
                task_name=task_name,
                gold_col=gold_col,
                pred_col=pred_col,
                output_dir=args.output_dir,
            )
        )

    write_markdown_report(args.output_dir, merged, results)

    print("Evaluation complete.")
    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()