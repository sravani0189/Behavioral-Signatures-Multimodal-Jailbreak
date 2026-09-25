#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    "llama32_11b_vision": "Llama-3.2-11B-Vision",
}

SETTINGS = {"original", "masked"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def normalize_model(model_id: str) -> str:
    return MODEL_DISPLAY.get(model_id.lower(), model_id)


def infer_variant_from_filename(name: str) -> str:
    upper = name.upper()
    if "SD_TYPO" in upper:
        return "SD_TYPO"
    if re.search(r"(?<![A-Z0-9])SD(?![A-Z0-9])", upper):
        return "SD"
    if re.search(r"(?<![A-Z0-9])TYPO(?![A-Z0-9])", upper):
        return "TYPO"
    return ""


def first_present(d: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in d and str(d[k]).strip() != "":
            return str(d[k]).strip()
    return default


def extract_question_meta(qmeta: Dict[str, Any], variant: str) -> Dict[str, str]:
    original_question = first_present(
        qmeta,
        ["Question", "original_question", "Original Question", "question"],
    )
    changed_question = first_present(
        qmeta,
        ["Changed Question", "changed_question", "rephrased_question", "prompt"],
    )
    key_phrase = first_present(
        qmeta,
        ["Key Phrase", "key_phrase", "phrase"],
    )
    phrase_type = first_present(
        qmeta,
        ["Phrase Type", "phrase_type", "type"],
    )

    if variant == "SD":
        prompt_source = "Rephrased Question(SD)"
        prompt = first_present(
            qmeta,
            ["Rephrased Question(SD)", "Rephrased Question", "prompt"],
        )
    else:
        prompt_source = "Rephrased Question"
        prompt = first_present(
            qmeta,
            ["Rephrased Question", "Rephrased Question(TYPO)", "prompt"],
        )

    return {
        "original_question": original_question,
        "changed_question": changed_question,
        "key_phrase": key_phrase,
        "phrase_type": phrase_type,
        "prompt_source": prompt_source,
        "prompt": prompt,
    }


def extract_response(row: Dict[str, Any]) -> str:
    return first_present(row, ["response", "text", "answer", "output"])


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    if "unsafe" in text:
        return "unsafe"
    if "safe" in text:
        return "safe"
    if "api_block" in text or "api block" in text or "blocked" in text:
        return "api_block"
    return ""


def extract_label(row: Dict[str, Any]) -> str:
    for key in [
        "is_safe(human_guideline_gpt)",
        "is_safe",
        "label",
        "safety_label",
        "prediction",
        "verdict",
    ]:
        if key in row and str(row[key]).strip() != "":
            return normalize_label(row[key])

    for val in row.values():
        lab = normalize_label(val)
        if lab:
            return lab

    return ""


def extract_qid(row: Dict[str, Any]) -> str:
    for key in ["question_id", "qid", "id"]:
        if key in row and str(row[key]).strip() != "":
            return str(row[key]).strip()
    return ""


def build_output_index(root: Path) -> Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Any]]]:
    """
    Key: (model_id, scenario, setting, variant)
    Value: {question_id: row}
    """
    index: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)

    out_root = root / "outputs"
    if not out_root.exists():
        raise FileNotFoundError(f"Missing outputs directory: {out_root}")

    for path in out_root.rglob("*_answers.jsonl"):
        rel = path.relative_to(out_root)
        parts = rel.parts
        if len(parts) < 4:
            continue

        model_id, scenario, setting = parts[0], parts[1], parts[2].lower()
        if setting not in SETTINGS:
            continue

        variant = infer_variant_from_filename(path.name)
        if not variant:
            continue

        key = (model_id, scenario, setting, variant)
        for row in read_jsonl(path):
            qid = extract_qid(row)
            if qid and qid not in index[key]:
                index[key][qid] = row

    return index


def build_judged_index(root: Path) -> Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Any]]]:
    """
    Key: (model_id, scenario, setting, variant)
    Value: {question_id: judged_row}
    """
    index: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)

    judged_root = root / "judged_results"
    if not judged_root.exists():
        raise FileNotFoundError(f"Missing judged_results directory: {judged_root}")

    for path in judged_root.rglob("*.json"):
        if path.name.endswith("_masked_human_guideline.json"):
            continue
        if path.name.endswith("_original_human_guideline.json"):
            continue

        rel = path.relative_to(judged_root)
        parts = rel.parts
        if len(parts) < 4:
            continue

        model_id, scenario, setting = parts[0], parts[1], parts[2].lower()
        if setting not in SETTINGS:
            continue

        variant = infer_variant_from_filename(path.name)
        if not variant:
            continue

        data = read_json(path)
        if not isinstance(data, list):
            continue

        key = (model_id, scenario, setting, variant)
        for row in data:
            if not isinstance(row, dict):
                continue
            qid = extract_qid(row)
            if qid and qid not in index[key]:
                index[key][qid] = row

    return index


def build_question_meta(root: Path, scenario: str) -> Dict[str, Dict[str, Any]]:
    qfile = root / "data" / "processed_questions" / f"{scenario}.json"
    if not qfile.exists():
        raise FileNotFoundError(f"Missing processed questions file: {qfile}")

    data = read_json(qfile)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {qfile}, got {type(data).__name__}")

    meta: Dict[str, Dict[str, Any]] = {}
    for qid, obj in data.items():
        if isinstance(obj, dict):
            meta[str(qid)] = obj
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="01-Illegal_Activitiy",
        help="Scenario name (matches data/processed_questions/<scenario>.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/behavior_analysis/data/behavior_pairs_raw.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    root = Path(".").resolve()

    question_meta = build_question_meta(root, args.scenario)
    output_index = build_output_index(root)
    judged_index = build_judged_index(root)

    rows: List[Dict[str, Any]] = []

    # Pair original/masked for the same model + scenario + variant + qid.
    pair_keys = set()
    for model_id, scenario, setting, variant in output_index.keys():
        if scenario == args.scenario and setting in SETTINGS:
            pair_keys.add((model_id, scenario, variant))

    if not pair_keys:
        raise RuntimeError(
            f"No output records found for scenario {args.scenario}. "
            "Check the outputs/<model>/<scenario>/<original|masked>/ layout."
        )

    for model_id, scenario, variant in sorted(pair_keys):
        orig_key = (model_id, scenario, "original", variant)
        mask_key = (model_id, scenario, "masked", variant)

        orig_rows = output_index.get(orig_key, {})
        mask_rows = output_index.get(mask_key, {})
        if not orig_rows or not mask_rows:
            continue

        common_qids = sorted(
            set(orig_rows.keys()) & set(mask_rows.keys()),
            key=lambda x: int(x) if str(x).isdigit() else x,
        )

        for qid in common_qids:
            qmeta = question_meta.get(qid, {})
            bench = extract_question_meta(qmeta, variant)

            orig_resp = extract_response(orig_rows[qid])
            mask_resp = extract_response(mask_rows[qid])

            orig_label = ""
            mask_label = ""

            if qid in judged_index.get(orig_key, {}):
                orig_label = extract_label(judged_index[orig_key][qid])
            if qid in judged_index.get(mask_key, {}):
                mask_label = extract_label(judged_index[mask_key][qid])

            transition = ""
            if orig_label and mask_label:
                transition = f"{orig_label.upper()} -> {mask_label.upper()}"

            sample_id = f"{model_id}_{scenario}_{variant}_{qid}"
            model_name = normalize_model(model_id)

            rows.append(
                {
                    "sample_id": sample_id,
                    "model": model_name,
                    "question_id": qid,
                    "scenario": scenario,
                    "variant": variant,
                    "original_question": bench["original_question"],
                    "changed_question": bench["changed_question"],
                    "key_phrase": bench["key_phrase"],
                    "phrase_type": bench["phrase_type"],
                    "prompt_source": bench["prompt_source"],
                    "prompt": bench["prompt"],
                    "original_response": orig_resp,
                    "masked_response": mask_resp,
                    "original_label": orig_label,
                    "masked_label": mask_label,
                    "transition": transition,
                }
            )

    rows.sort(
        key=lambda r: (
            r["model"],
            r["scenario"],
            r["variant"],
            int(r["question_id"]) if str(r["question_id"]).isdigit() else r["question_id"],
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "model",
                "question_id",
                "scenario",
                "variant",
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
                "transition",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {args.output}")
    print(f"[OK] Rows: {len(rows)}")


if __name__ == "__main__":
    main()