from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# Map GPT-labeled files to human-annotated files
# ---------------------------------------------------------------------
PAIRS = [
    {
        "dataset_name": "Illegal_Activity_Image",
        "gpt_file": "results/01-Illegal_Activitiy_SD_labeled.csv",
        "human_file": "results/01-Illegal_Activitiy_Image_human_annotation.xlsx",
    },
    {
        "dataset_name": "Illegal_Activity_Image_Text",
        "gpt_file": "results/01-Illegal_Activitiy_SD_TYPO_labeled.csv",
        "human_file": "results/01-Illegal_Activitiy_Image_Text_human_annotation.xlsx",
    },
    {
        "dataset_name": "Illegal_Activity_Text",
        "gpt_file": "results/01-Illegal_Activitiy_TYPO_labeled.csv",
        "human_file": "results/01-Illegal_Activitiy_Text_human_annotation.xlsx",
    },
]

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


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
    """Map SAFE/UNSAFE to binary labels. UNSAFE is the positive class."""
    if label == "UNSAFE":
        return 1
    if label == "SAFE":
        return 0
    return None


def compute_metrics(y_true, y_pred):
    """Compute confusion matrix and unsafe-class metrics."""
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

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


all_summary = []
all_disagreements = []

for item in PAIRS:
    dataset_name = item["dataset_name"]
    gpt_path = Path(item["gpt_file"])
    human_path = Path(item["human_file"])

    print(f"\n=== {dataset_name} ===")
    print(f"GPT file  : {gpt_path}")
    print(f"Human file: {human_path}")

    if not gpt_path.exists():
        print(f"Missing GPT file: {gpt_path}")
        continue
    if not human_path.exists():
        print(f"Missing human file: {human_path}")
        continue

    gpt_df = pd.read_csv(gpt_path)
    human_df = pd.read_excel(human_path)

    if "question_id" not in gpt_df.columns or "label" not in gpt_df.columns:
        print(f"ERROR: GPT file {gpt_path.name} must contain question_id and label")
        continue

    if "question_id" not in human_df.columns or "Human_Label" not in human_df.columns:
        print(f"ERROR: Human file {human_path.name} must contain question_id and Human_Label")
        continue

    # Normalize labels
    gpt_df["GPT_Label_Norm"] = gpt_df["label"].apply(normalize_label)
    human_df["Human_Label_Norm"] = human_df["Human_Label"].apply(normalize_label)

    # Merge on question_id
    merged = pd.merge(
        gpt_df[["question_id", "response", "GPT_Label_Norm"]],
        human_df[["question_id", "Human_Label_Norm", "Notes"]],
        on="question_id",
        how="inner",
        validate="one_to_one",
    )

    # Convert labels to binary
    merged["gpt_bin"] = merged["GPT_Label_Norm"].apply(to_binary)
    merged["human_bin"] = merged["Human_Label_Norm"].apply(to_binary)

    valid = merged.dropna(subset=["gpt_bin", "human_bin"]).copy()
    valid["gpt_bin"] = valid["gpt_bin"].astype(int)
    valid["human_bin"] = valid["human_bin"].astype(int)

    if len(valid) == 0:
        print("No valid rows after filtering.")
        continue

    metrics = compute_metrics(valid["human_bin"].tolist(), valid["gpt_bin"].tolist())

    summary_row = {
        "Dataset": dataset_name,
        "Total_Matched_Rows": len(merged),
        "Total_Valid_Rows": len(valid),
        "Human_SAFE": int((valid["human_bin"] == 0).sum()),
        "Human_UNSAFE": int((valid["human_bin"] == 1).sum()),
        "GPT_SAFE": int((valid["gpt_bin"] == 0).sum()),
        "GPT_UNSAFE": int((valid["gpt_bin"] == 1).sum()),
        **metrics,
    }
    all_summary.append(summary_row)

    # Save disagreements for error analysis
    disagreements = valid[valid["gpt_bin"] != valid["human_bin"]].copy()
    disagreements["Dataset"] = dataset_name
    disagreements = disagreements[
        ["Dataset", "question_id", "response", "GPT_Label_Norm", "Human_Label_Norm", "Notes"]
    ]
    all_disagreements.append(disagreements)

    disagreement_path = OUTPUT_DIR / f"{dataset_name}_gpt_vs_human_disagreements.csv"
    disagreements.to_csv(disagreement_path, index=False)
    print(f"Saved disagreements -> {disagreement_path.name}")

# ---------------------------------------------------------------------
# Save summary
# ---------------------------------------------------------------------
summary_df = pd.DataFrame(all_summary)

print("\n==============================")
print("GPT vs Human Summary")
print("==============================")
print(summary_df)

summary_out = OUTPUT_DIR / "GPT_vs_Human_Metrics.xlsx"
summary_df.to_excel(summary_out, index=False)
print(f"\nSaved summary -> {summary_out}")

# ---------------------------------------------------------------------
# Save combined disagreements
# ---------------------------------------------------------------------
if all_disagreements:
    combined_disagreements = pd.concat(all_disagreements, ignore_index=True)
    combined_out = OUTPUT_DIR / "GPT_vs_Human_All_Disagreements.csv"
    combined_disagreements.to_csv(combined_out, index=False)
    print(f"Saved combined disagreements -> {combined_out}")
else:
    print("No disagreements found.")