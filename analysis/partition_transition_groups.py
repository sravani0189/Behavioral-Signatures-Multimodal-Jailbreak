from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FILES = [
    "experiments/ocr_masking_transition/outputs/01-Illegal_Activitiy_SD_transition_table.csv",
    "experiments/ocr_masking_transition/outputs/01-Illegal_Activitiy_SD_TYPO_transition_table.csv",
    "experiments/ocr_masking_transition/outputs/01-Illegal_Activitiy_TYPO_transition_table.csv",
]

ORIGINAL_FILE = Path("data/processed_questions/01-Illegal_Activitiy.json")
OUT_DIR = Path("analysis/transition_groups")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_PROMPT_KEY = {
    "SD": "Rephrased Question(SD)",
    "SD_TYPO": "Rephrased Question",
    "TYPO": "Rephrased Question",
}


def load_original_context(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_prompt_context(item: dict, variant: str) -> dict:
    prompt_key = VARIANT_PROMPT_KEY[variant]

    return {
        "original_question": item.get("Question", ""),
        "changed_question": item.get("Changed Question", ""),
        "key_phrase": item.get("Key Phrase", ""),
        "phrase_type": item.get("Phrase Type", ""),
        "rephrased_question": item.get("Rephrased Question", ""),
        "rephrased_question_sd": item.get("Rephrased Question(SD)", ""),
        "prompt": item.get(prompt_key, ""),
        "prompt_source": prompt_key,
    }


def main() -> None:
    original_data = load_original_context(ORIGINAL_FILE)

    dfs = [pd.read_csv(f) for f in FILES]
    all_df = pd.concat(dfs, ignore_index=True)

    # In the existing transition tables, the column named "question" is actually
    # the prompt sent to LLaVA. Rename it for clarity.
    all_df = all_df.rename(columns={"question": "prompt"})

    # Add benchmark context fields from the original MM-SafetyBench JSON.
    enriched_rows = []
    for _, row in all_df.iterrows():
        qid = str(int(row["question_id"])) if str(row["question_id"]).isdigit() else str(row["question_id"])
        variant = str(row["variant"])

        if qid not in original_data:
            raise KeyError(f"question_id not found in original benchmark file: {qid}")

        item = original_data[qid]
        context = get_prompt_context(item, variant)

        enriched_row = row.to_dict()
        enriched_row.update(context)
        enriched_rows.append(enriched_row)

    all_df = pd.DataFrame(enriched_rows)

    all_df["transition"] = all_df["transition"].astype(str).str.strip()

    # A clean column order for analysis.
    preferred_cols = [
        "question_id",
        "variant",
        "original_question",
        "changed_question",
        "key_phrase",
        "phrase_type",
        "rephrased_question",
        "rephrased_question_sd",
        "prompt_source",
        "prompt",
        "original_label",
        "masked_label",
        "transition",
        "changed",
        "original_response",
        "masked_response",
        "image_path",
    ]
    cols = [c for c in preferred_cols if c in all_df.columns] + [
        c for c in all_df.columns if c not in preferred_cols
    ]
    all_df = all_df[cols]

    transition_map = {
        "SAFE -> SAFE": "SAFE_to_SAFE.csv",
        "SAFE -> UNSAFE": "SAFE_to_UNSAFE.csv",
        "UNSAFE -> SAFE": "UNSAFE_to_SAFE.csv",
        "UNSAFE -> UNSAFE": "UNSAFE_to_UNSAFE.csv",
    }

    for transition, filename in transition_map.items():
        sub = all_df[all_df["transition"] == transition].copy()
        out_path = OUT_DIR / filename
        sub.to_csv(out_path, index=False)
        print(f"{transition}: {len(sub)} samples -> {filename}")

    summary = (
        all_df.groupby(["variant", "transition"])
        .size()
        .reset_index(name="count")
        .sort_values(["variant", "transition"])
    )

    summary_path = OUT_DIR / "transition_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved summary ->", summary_path)
    print("\nOverall counts:")
    print(all_df["transition"].value_counts().to_string())


if __name__ == "__main__":
    main()