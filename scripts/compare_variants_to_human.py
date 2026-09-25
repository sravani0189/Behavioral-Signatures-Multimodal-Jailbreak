from pathlib import Path
import json
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
JSON_FILE = Path("questions_with_answers/01-Illegal_Activitiy_human_guideline.json")

HUMAN_FILES = {
    "SD": Path("results/01-Illegal_Activitiy_Image_human_annotation.xlsx"),
    "SD_TYPO": Path("results/01-Illegal_Activitiy_Image_Text_human_annotation.xlsx"),
    "TYPO": Path("results/01-Illegal_Activitiy_Text_human_annotation.xlsx"),
}

VARIANT_NAME = {
    "SD": "Image",
    "SD_TYPO": "Image_Text",
    "TYPO": "Text",
}

LABEL_FIELDS = {
    "Original_GPT": "is_safe(gpt)",
    "Human_Guideline_GPT": "is_safe(human_guideline_gpt)",
}

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_label(value):
    """Normalize labels like 'SAFE\\nConfidence: High' -> 'SAFE'."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip().upper()
    if first_line in {"SAFE", "UNSAFE"}:
        return first_line
    return first_line


def to_binary(label):
    """SAFE -> 0, UNSAFE -> 1."""
    if label == "SAFE":
        return 0
    if label == "UNSAFE":
        return 1
    return None


def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, F1 for UNSAFE as the positive class.
    """
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
    }


def load_human_df(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {"question_id", "Human_Label", "Notes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    df = df.copy()
    df["question_id"] = df["question_id"].astype(str).str.strip()
    df["Human_Label_Norm"] = df["Human_Label"].apply(normalize_label)
    df["Human_Bin"] = df["Human_Label_Norm"].apply(to_binary)
    return df


def load_json_rows(path: Path) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)

    rows = []
    for qid, item in data.items():
        qid_str = str(qid).strip()
        ans = item.get("ans", {})
        for variant_key in ["SD", "SD_TYPO", "TYPO"]:
            if variant_key not in ans:
                continue
            row = {
                "question_id": qid_str,
                "variant_key": variant_key,
                "response": ans[variant_key].get("text", ""),
            }
            for judge_name, field_name in LABEL_FIELDS.items():
                row[f"{judge_name}_Label"] = ans[variant_key].get(field_name, "")
                row[f"{judge_name}_Norm"] = normalize_label(row[f"{judge_name}_Label"])
                row[f"{judge_name}_Bin"] = to_binary(row[f"{judge_name}_Norm"])
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------
def main():
    if not JSON_FILE.exists():
        raise FileNotFoundError(f"Could not find {JSON_FILE}")

    json_df = load_json_rows(JSON_FILE)

    summary_rows = []
    all_disagreements = []

    for variant_key, human_path in HUMAN_FILES.items():
        variant_name = VARIANT_NAME[variant_key]
        print(f"\n=== {variant_name} ===")
        print(f"Human file: {human_path.name}")

        human_df = load_human_df(human_path)

        variant_json = json_df[json_df["variant_key"] == variant_key].copy()

        merged = pd.merge(
            variant_json,
            human_df[["question_id", "Human_Label_Norm", "Human_Bin", "Notes"]],
            on="question_id",
            how="inner",
            validate="one_to_one",
        )

        if len(merged) == 0:
            print("No matched rows.")
            continue

        for judge_name in LABEL_FIELDS.keys():
            pred_col = f"{judge_name}_Bin"

            valid = merged.dropna(subset=[pred_col, "Human_Bin"]).copy()
            valid[pred_col] = valid[pred_col].astype(int)
            valid["Human_Bin"] = valid["Human_Bin"].astype(int)

            metrics = compute_metrics(valid["Human_Bin"].tolist(), valid[pred_col].tolist())

            summary_rows.append({
                "Variant": variant_name,
                "Judge": judge_name,
                "Matched_Rows": len(valid),
                "Human_SAFE": int((valid["Human_Bin"] == 0).sum()),
                "Human_UNSAFE": int((valid["Human_Bin"] == 1).sum()),
                "Pred_SAFE": int((valid[pred_col] == 0).sum()),
                "Pred_UNSAFE": int((valid[pred_col] == 1).sum()),
                **metrics,
            })

            disagreements = valid[valid[pred_col] != valid["Human_Bin"]].copy()
            disagreements["Variant"] = variant_name
            disagreements["Judge"] = judge_name
            disagreements = disagreements[[
                "Variant", "Judge", "question_id", "response",
                f"{judge_name}_Norm", "Human_Label_Norm", "Notes"
            ]]

            out_csv = OUTPUT_DIR / f"{variant_name}_{judge_name}_disagreements.csv"
            disagreements.to_csv(out_csv, index=False)
            print(f"{judge_name}: saved disagreements -> {out_csv.name}")

            all_disagreements.append(disagreements)

    summary_df = pd.DataFrame(summary_rows)

    print("\n==============================")
    print("GPT vs Human Summary")
    print("==============================")
    print(summary_df)

    summary_out = OUTPUT_DIR / "Judge_Comparison_To_Human.xlsx"
    summary_df.to_excel(summary_out, index=False)
    print(f"\nSaved summary -> {summary_out}")

    if all_disagreements:
        combined = pd.concat(all_disagreements, ignore_index=True)
        combined_out = OUTPUT_DIR / "Judge_Comparison_All_Disagreements.csv"
        combined.to_csv(combined_out, index=False)
        print(f"Saved combined disagreements -> {combined_out}")


if __name__ == "__main__":
    main()