import json
import csv
from pathlib import Path

ORIGINAL_FILE = Path("eval_results/01-Illegal_Activitiy_human_guideline.json")
MASKED_FILE = Path("judged_results/masked_human_guideline_summary.csv")

OUT_CSV = Path("results/before_after_comparison.csv")
OUT_JSON = Path("results/before_after_comparison.json")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load original results
# -----------------------------
with open(ORIGINAL_FILE, "r", encoding="utf-8") as f:
    original = json.load(f)

# -----------------------------
# Load masked results
# -----------------------------
masked = {}
with open(MASKED_FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row["file"]

        if "SD_TYPO" in fname:
            key = "SD_TYPO"
        elif "TYPO" in fname and "SD_TYPO" not in fname:
            key = "TYPO"
        elif "SD_answers" in fname or "_SD_" in fname or fname.endswith("SD_answers.jsonl"):
            key = "SD"
        else:
            continue

        masked[key] = {
            "safe": int(row["safe"]),
            "unsafe": int(row["unsafe"]),
            "attack_rate": float(row["attack_rate"]),
        }

# -----------------------------
# Compare
# -----------------------------
rows = []
summary = {}

for variant in ["SD", "SD_TYPO", "TYPO"]:
    before = original[variant]
    after = masked[variant]

    before_safe = int(before["safe"])
    before_unsafe = int(before["unsafe"])
    before_rate = float(before["attack_rate"])

    after_safe = int(after["safe"])
    after_unsafe = int(after["unsafe"])
    after_rate = float(after["attack_rate"])

    unsafe_change = after_unsafe - before_unsafe
    rate_change = after_rate - before_rate

    row = {
        "Variant": variant,
        "Before_SAFE": before_safe,
        "Before_UNSAFE": before_unsafe,
        "Before_Attack_Rate": round(before_rate, 4),
        "After_SAFE": after_safe,
        "After_UNSAFE": after_unsafe,
        "After_Attack_Rate": round(after_rate, 4),
        "Unsafe_Change": unsafe_change,
        "Attack_Rate_Change": round(rate_change, 4),
    }
    rows.append(row)
    summary[variant] = row

# -----------------------------
# Save CSV
# -----------------------------
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# -----------------------------
# Save JSON
# -----------------------------
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# -----------------------------
# Print results
# -----------------------------
print("\nBefore vs After OCR Masking\n")
for r in rows:
    print(f"{r['Variant']}:")
    print(f"  Before: {r['Before_SAFE']} safe / {r['Before_UNSAFE']} unsafe ({r['Before_Attack_Rate']:.4f})")
    print(f"  After : {r['After_SAFE']} safe / {r['After_UNSAFE']} unsafe ({r['After_Attack_Rate']:.4f})")
    print(f"  Change: {r['Unsafe_Change']:+d} unsafe, {r['Attack_Rate_Change']:+.4f} attack-rate change\n")

print(f"Saved -> {OUT_CSV}")
print(f"Saved -> {OUT_JSON}")