from pathlib import Path
import pandas as pd

FILES = [
    "results/01-Illegal_Activitiy_Image_human_annotation.xlsx",
    "results/01-Illegal_Activitiy_Image_Text_human_annotation.xlsx",
    "results/01-Illegal_Activitiy_Text_human_annotation.xlsx",
]

summary = []

for file in FILES:
    path = Path(file)
    print(f"\nReading {path.name}")

    df = pd.read_excel(path)

    required_cols = ["question_id", "GPT_Label", "response", "Human_Label", "Notes"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in {path.name}: {missing}")
        continue

    # Extract only the first line of Human_Label
    # Example: "SAFE\nConfidence: High" -> "SAFE"
    raw_labels = df["Human_Label"].fillna("").astype(str).str.strip()
    cleaned_labels = raw_labels.apply(lambda x: x.splitlines()[0].strip().upper() if x.strip() else "")

    total_rows = len(df)
    annotated_rows = (cleaned_labels != "").sum()
    safe_count = (cleaned_labels == "SAFE").sum()
    unsafe_count = (cleaned_labels == "UNSAFE").sum()
    blank_count = (cleaned_labels == "").sum()

    unsafe_rate = round((unsafe_count / annotated_rows) * 100, 2) if annotated_rows else 0.0

    summary.append({
        "Dataset": path.stem.replace("_human_annotation", ""),
        "Total Rows": total_rows,
        "Annotated Rows": annotated_rows,
        "Blank Rows": blank_count,
        "SAFE": safe_count,
        "UNSAFE": unsafe_count,
        "Unsafe Rate (%)": unsafe_rate,
    })

    # Save a cleaned copy
    df["Human_Label"] = cleaned_labels
    cleaned_out = path.with_name(path.stem.replace("_human_annotation", "") + "_cleaned.xlsx")
    df.to_excel(cleaned_out, index=False)
    print(f"Saved cleaned file -> {cleaned_out.name}")

summary_df = pd.DataFrame(summary)

print("\n==============================")
print("Human Annotation Summary")
print("==============================")
print(summary_df)

summary_out = Path("results/Human_Annotation_Summary.xlsx")
summary_df.to_excel(summary_out, index=False)
print(f"\nSummary saved to -> {summary_out}")