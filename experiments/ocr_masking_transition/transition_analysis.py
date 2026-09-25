from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


SCENARIO = "01-Illegal_Activitiy"
VARIANTS = ["SD", "SD_TYPO", "TYPO"]

DEFAULT_ORIGINAL = Path("questions_with_answers/01-Illegal_Activitiy_human_guideline.json")
DEFAULT_MASKED_DIR = Path("judged_results")
DEFAULT_OUT_DIR = Path("experiments/ocr_masking_transition/outputs")

LABEL_FIELD = "is_safe(human_guideline_gpt)"


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"safe", "unsafe"}:
        return text
    match = re.search(r"\b(safe|unsafe)\b", text)
    return match.group(1) if match else text


def load_original(original_path: Path) -> Dict[str, Any]:
    if not original_path.exists():
        raise FileNotFoundError(f"Missing original file: {original_path}")
    with original_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_masked(masked_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load masked JSONL rows keyed by question/sample id.

    The masked rows may contain any of these keys:
    question_id, id, qid, sample_id, index.
    If no id field is present, we fall back to line order.
    """
    if not masked_path.exists():
        raise FileNotFoundError(f"Missing masked file: {masked_path}")

    rows: Dict[str, Dict[str, Any]] = {}
    with masked_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            qid = None
            for key in ("question_id", "id", "qid", "sample_id", "index"):
                if key in obj and obj[key] is not None:
                    qid = str(obj[key])
                    break
            if qid is None:
                qid = str(idx)

            rows[qid] = obj
    return rows


def extract_original_label(original_data: Dict[str, Any], qid: str, variant: str) -> str:
    item = original_data.get(qid)
    if item is None:
        raise KeyError(f"Original qid not found: {qid}")

    ans = item.get("ans", {})
    if variant not in ans:
        raise KeyError(f"Variant {variant} missing for qid={qid}")

    answer_obj = ans[variant]
    if LABEL_FIELD not in answer_obj:
        raise KeyError(f"Label field {LABEL_FIELD!r} missing for qid={qid}, variant={variant}")

    return normalize_label(answer_obj[LABEL_FIELD])


def extract_masked_label(masked_row: Dict[str, Any]) -> str:
    if LABEL_FIELD in masked_row:
        return normalize_label(masked_row[LABEL_FIELD])

    for key in ("label", "is_safe", "judge_label", "human_guideline_label"):
        if key in masked_row:
            return normalize_label(masked_row[key])

    raise KeyError("Could not find a label field in masked row.")


def extract_response_text(row: Dict[str, Any]) -> str:
    for key in ("response", "text", "answer", "output", "model_response", "masked_response", "generated_text"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def extract_question_text(item: Dict[str, Any]) -> str:
    for key in ("question", "Question", "query", "prompt"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def transition(before: str, after: str) -> str:
    return f"{before.upper()} -> {after.upper()}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build before/after transition tables for OCR masking."
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=DEFAULT_ORIGINAL,
        help="Original human-guideline JSON file with per-sample labels.",
    )
    parser.add_argument(
        "--masked-dir",
        type=Path,
        default=DEFAULT_MASKED_DIR,
        help="Directory containing masked judged JSONL files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for transition tables.",
    )
    args = parser.parse_args()

    original_data = load_original(args.original)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for variant in VARIANTS:
        masked_file = args.masked_dir / f"{SCENARIO}_{variant}_human_guideline_judged.jsonl"

        masked_rows = load_masked(masked_file)

        variant_rows: List[Dict[str, Any]] = []
        transition_counts = {
            "SAFE -> SAFE": 0,
            "SAFE -> UNSAFE": 0,
            "UNSAFE -> SAFE": 0,
            "UNSAFE -> UNSAFE": 0,
        }

        qids = sorted(original_data.keys(), key=lambda x: int(x))
        total = 0
        changed = 0
        missing = 0

        for qid in qids:
            original_item = original_data[qid]
            original_answer = original_item.get("ans", {}).get(variant, {})

            if qid not in masked_rows:
                missing += 1
                variant_rows.append(
                    {
                        "question_id": int(qid),
                        "variant": variant,
                        "question": extract_question_text(original_item),
                        "image_path": original_item.get("image_path", ""),
                        "original_label": extract_original_label(original_data, qid, variant),
                        "masked_label": "",
                        "transition": "MISSING_MASKED_ROW",
                        "changed": "",
                        "original_response": extract_response_text(original_answer),
                        "masked_response": "",
                    }
                )
                continue

            before_label = extract_original_label(original_data, qid, variant)
            masked_row = masked_rows[qid]
            after_label = extract_masked_label(masked_row)

            t = transition(before_label, after_label)
            transition_counts[t] = transition_counts.get(t, 0) + 1
            total += 1
            if before_label != after_label:
                changed += 1

            variant_rows.append(
                {
                    "question_id": int(qid),
                    "variant": variant,
                    "question": extract_question_text(original_item),
                    "image_path": original_item.get("image_path", ""),
                    "original_label": before_label,
                    "masked_label": after_label,
                    "transition": t,
                    "changed": before_label != after_label,
                    "original_response": extract_response_text(original_answer),
                    "masked_response": extract_response_text(masked_row),
                }
            )

        csv_path = args.out_dir / f"{SCENARIO}_{variant}_transition_table.csv"
        jsonl_path = args.out_dir / f"{SCENARIO}_{variant}_transition_table.jsonl"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "question_id",
                    "variant",
                    "question",
                    "image_path",
                    "original_label",
                    "masked_label",
                    "transition",
                    "changed",
                    "original_response",
                    "masked_response",
                ],
            )
            writer.writeheader()
            writer.writerows(variant_rows)

        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in variant_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary_rows.append(
            {
                "variant": variant,
                "total_compared": total,
                "changed": changed,
                "unchanged": total - changed,
                "missing_masked_rows": missing,
                "change_rate": round(changed / total if total else 0.0, 4),
                "safe_to_safe": transition_counts.get("SAFE -> SAFE", 0),
                "safe_to_unsafe": transition_counts.get("SAFE -> UNSAFE", 0),
                "unsafe_to_safe": transition_counts.get("UNSAFE -> SAFE", 0),
                "unsafe_to_unsafe": transition_counts.get("UNSAFE -> UNSAFE", 0),
                "masked_file": str(masked_file),
            }
        )

        print(
            f"{variant}: total={total}, changed={changed}, "
            f"change_rate={changed / total if total else 0.0:.4f}, missing={missing}"
        )
        print(
            "  transitions:",
            f"S->S={transition_counts.get('SAFE -> SAFE', 0)}",
            f"S->U={transition_counts.get('SAFE -> UNSAFE', 0)}",
            f"U->S={transition_counts.get('UNSAFE -> SAFE', 0)}",
            f"U->U={transition_counts.get('UNSAFE -> UNSAFE', 0)}",
        )

    summary_json = args.out_dir / f"{SCENARIO}_transition_summary.json"
    summary_csv = args.out_dir / f"{SCENARIO}_transition_summary.csv"

    summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "total_compared",
                "changed",
                "unchanged",
                "missing_masked_rows",
                "change_rate",
                "safe_to_safe",
                "safe_to_unsafe",
                "unsafe_to_safe",
                "unsafe_to_unsafe",
                "masked_file",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("Saved:")
    print(summary_json)
    print(summary_csv)


if __name__ == "__main__":
    main()