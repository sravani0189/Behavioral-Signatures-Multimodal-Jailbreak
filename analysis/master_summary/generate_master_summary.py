#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


MODEL_DISPLAY = {
    "qwen25_vl": "Qwen2.5-VL",
    "internvl25": "InternVL2.5",
    "molmo": "Molmo-7B",
    "gpt56_sol": "GPT-5.6 Sol",
    "gpt56_terra": "GPT-5.6 Terra",
    "phi4mm": "Phi-4 Multimodal",
    "minicpmv26": "MiniCPM-V 2.6",
    "pixtral12b": "Pixtral-12B",
    "ovis25_9b": "Ovis2.5-9B",
    "llama32_vision": "Llama-3.2-11B-Vision",
}

VALID_MODELS = {
    "Qwen2.5-VL",
    "InternVL2.5",
    "Molmo-7B",
    "GPT-5.6 Sol",
    "GPT-5.6 Terra",
    "Phi-4 Multimodal",
    "MiniCPM-V 2.6",
    "Pixtral-12B",
    "Ovis2.5-9B",
    "Llama-3.2-11B-Vision",
}

OUTPUT_COLUMNS = [
    "model",
    "scenario",
    "setting",
    "variant",
    "safe",
    "unsafe",
    "api_block",
    "total",
    "attack_rate",
    "blocked_rate",
    "evaluated_attack_rate",
    "source_file",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_model(folder_name: str) -> str:
    return MODEL_DISPLAY.get(folder_name.lower(), folder_name)


def as_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value
    return value


def find_first_key_value(obj: Any, keys: Iterable[str]) -> Optional[Any]:
    key_set = set(keys)

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in key_set:
                return v
        for v in obj.values():
            found = find_first_key_value(v, keys)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = find_first_key_value(item, keys)
            if found is not None:
                return found

    return None


def extract_metrics(metric_obj: Dict[str, Any]) -> Dict[str, Any]:
    safe = as_number(find_first_key_value(metric_obj, ["safe", "safe_count", "n_safe"])) or 0
    unsafe = as_number(find_first_key_value(metric_obj, ["unsafe", "unsafe_count", "n_unsafe"])) or 0
    api_block = as_number(
        find_first_key_value(metric_obj, ["api_block", "api_block_count", "blocked", "blocked_count"])
    ) or 0

    total = as_number(find_first_key_value(metric_obj, ["total", "n_total"]))
    if total is None:
        total = safe + unsafe + api_block

    attack_rate = find_first_key_value(metric_obj, ["attack_rate", "attack_pct"])
    blocked_rate = find_first_key_value(metric_obj, ["blocked_rate", "blocked_pct"])
    evaluated_attack_rate = find_first_key_value(
        metric_obj, ["evaluated_attack_rate", "evaluated_attack_pct"]
    )

    if attack_rate is None:
        attack_rate = (unsafe / total) if total else 0.0
    if blocked_rate is None:
        blocked_rate = (api_block / total) if total else 0.0
    if evaluated_attack_rate is None:
        denom = safe + unsafe
        evaluated_attack_rate = (unsafe / denom) if denom else 0.0

    return {
        "safe": int(safe),
        "unsafe": int(unsafe),
        "api_block": int(api_block),
        "total": int(total),
        "attack_rate": attack_rate,
        "blocked_rate": blocked_rate,
        "evaluated_attack_rate": evaluated_attack_rate,
    }


def infer_setting(path: Path) -> str:
    name = path.name.lower()
    if "masked" in name:
        return "Masked"
    return "Original"


def infer_variant_from_key(key: str) -> str:
    text = key.upper()

    if "SD_TYPO" in text:
        return "SD_TYPO"
    if re.search(r"(?<![A-Z0-9])SD(?![A-Z0-9])", text):
        return "SD"
    if re.search(r"(?<![A-Z0-9])TYPO(?![A-Z0-9])", text):
        return "TYPO"
    return "ALL"


def infer_scenario_from_key(key: str) -> str:
    stem = key
    stem = re.sub(r"_human_guideline(_masked)?$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(SD_TYPO|SD|TYPO)_answers$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_answers$", "", stem, flags=re.IGNORECASE)
    return stem


def infer_model(eval_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(eval_root)
    if len(rel.parts) < 2:
        return "SKIP"
    model_folder = rel.parts[0]
    return normalize_model(model_folder)


def should_include(file_path: Path) -> bool:
    name = file_path.name.lower()
    if not name.endswith(".json"):
        return False
    return "_human_guideline" in name


def file_has_nested_metrics(obj: Any) -> bool:
    return isinstance(obj, dict) and any(isinstance(v, dict) for v in obj.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("eval_results"),
        help="Root folder containing evaluation JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("analysis/master_summary/master_summary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("analysis/master_summary/master_summary.md"),
        help="Output Markdown path.",
    )
    args = parser.parse_args()

    eval_root = args.eval_root
    if not eval_root.exists():
        raise FileNotFoundError(f"eval_root does not exist: {eval_root}")

    rows = []

    for json_path in sorted(eval_root.rglob("*.json")):
        if not should_include(json_path):
            continue

        model = infer_model(eval_root, json_path)
        if model == "SKIP" or model not in VALID_MODELS:
            continue

        try:
            data = read_json(json_path)
        except Exception as e:
            print(f"[WARN] Skipping unreadable file: {json_path} ({e})")
            continue

        if not file_has_nested_metrics(data):
            continue

        for key, metric_obj in data.items():
            if not isinstance(metric_obj, dict):
                continue

            metrics = extract_metrics(metric_obj)

            row = {
                "model": model,
                "scenario": infer_scenario_from_key(key),
                "setting": infer_setting(json_path),
                "variant": infer_variant_from_key(key),
                "safe": metrics["safe"],
                "unsafe": metrics["unsafe"],
                "api_block": metrics["api_block"],
                "total": metrics["total"],
                "attack_rate": metrics["attack_rate"],
                "blocked_rate": metrics["blocked_rate"],
                "evaluated_attack_rate": metrics["evaluated_attack_rate"],
                "source_file": str(json_path.relative_to(eval_root)),
            }
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No metric JSON files found under {eval_root}. "
            "Check that files contain nested human_guideline metrics and are inside model folders."
        )

    rows.sort(key=lambda r: (r["model"], r["scenario"], r["setting"], r["variant"], r["source_file"]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with args.output_md.open("w", encoding="utf-8") as f:
        f.write("# Master Summary\n\n")
        f.write(f"Total rows: {len(rows)}\n\n")

        current_model = None
        for row in rows:
            if row["model"] != current_model:
                current_model = row["model"]
                f.write(f"## {current_model}\n\n")
                f.write(
                    "| Scenario | Setting | Variant | Safe | Unsafe | API Block | Total | Attack Rate | Blocked Rate | Evaluated Attack Rate | Source |\n"
                )
                f.write(
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
                )

            f.write(
                f"| {row['scenario']} | {row['setting']} | {row['variant']} | "
                f"{row['safe']} | {row['unsafe']} | {row['api_block']} | {row['total']} | "
                f"{row['attack_rate']} | {row['blocked_rate']} | {row['evaluated_attack_rate']} | "
                f"{row['source_file']} |\n"
            )

    print(f"[OK] Wrote {args.output_csv}")
    print(f"[OK] Wrote {args.output_md}")
    print(f"[OK] Rows: {len(rows)}")


if __name__ == "__main__":
    main()