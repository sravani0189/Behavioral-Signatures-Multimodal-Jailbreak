import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# Paper-style formatting
# ------------------------
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ------------------------
# Load data
# ------------------------
input_path = "analysis/behavior_analysis/outputs/fewshot/behavioral_signature_matrix.csv"
output_path = "figure4_behavior_distribution.pdf"

df = pd.read_csv(input_path, index_col=0)
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

# ------------------------
# Rename models for paper display
# ------------------------
rename_models = {
    "GPT-5.6 Sol": "GPT-5.6 Sol",
    "GPT-5.6 Terra": "GPT-5.6 Terra",
    "InternVL2.5": "InternVL2.5",
    "MiniCPM-V 2.6": "MiniCPM",
    "Molmo-7B": "Molmo",
    "Ovis2.5-9B": "Ovis2.5",
    "Phi-4 Multimodal": "Phi-4 MM",
    "Pixtral-12B": "Pixtral",
    "Qwen2.5-VL": "Qwen2.5",
    "llama32_vision": "Llama-3.2"
}
df.index = df.index.map(lambda x: rename_models.get(x, x))

# ------------------------
# Order models by family
# ------------------------
model_order = [
    "GPT-5.6 Sol",
    "GPT-5.6 Terra",
    "InternVL2.5",
    "Phi-4 MM",
    "Llama-3.2",
    "Ovis2.5",
    "Qwen2.5",
    "MiniCPM",
    "Molmo",
    "Pixtral"
]
df = df.reindex([m for m in model_order if m in df.index])

# ------------------------
# Order mechanisms
# ------------------------
column_order = [
    "Stable Refusal",
    "Visual Grounding",
    "Operational Reinterpretation",
    "OCR Dependency",
    "Image Insufficient",
    "Visual Insufficient",
    "Prompt Dependency",
    "Ethical Consideration",
    "Harmful Intent Ignored",
    "Benign Reinterpretation"
]
df = df[[c for c in column_order if c in df.columns]]

# ------------------------
# Colors: muted paper-style palette
# ------------------------
colors = {
    "Stable Refusal": "#8FA8C4",
    "Visual Grounding": "#C9D89B",
    "Operational Reinterpretation": "#C7A08B",
    "OCR Dependency": "#C9A7D0",
    "Image Insufficient": "#E3A6A1",
    "Visual Insufficient": "#9ED6E0",
    "Prompt Dependency": "#D8C2A6",
    "Ethical Consideration": "#B8D8D8",
    "Harmful Intent Ignored": "#F0D48A",
    "Benign Reinterpretation": "#D5D5D5"
}
color_list = [colors[c] for c in df.columns]

# ------------------------
# Diagnostics
# ------------------------
print("Shape:", df.shape)
print("Row sums:")
print(df.sum(axis=1))
print("Dominant category per model:")
print(df.idxmax(axis=1))

# ------------------------
# Plot
# ------------------------
fig, ax = plt.subplots(figsize=(7.4, 3.9))

df.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    width=0.58,
    color=color_list,
    edgecolor="white",
    linewidth=0.25
)

ax.set_ylabel("Percentage (%)")
ax.set_xlabel("")
ax.set_title("")
ax.set_ylim(0, 100)

# Ticks and grid
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.tick_params(axis="x", rotation=22, length=0)
for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")

ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.18)
ax.set_axisbelow(True)
ax.margins(x=0.03)

# Family separators
# After index 4 and 6 in the ordered list:
# [F0: 0-4], [F1: 5-6], [F2: 7-9]
ax.axvline(4.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
ax.axvline(6.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)

# Compact horizontal legend
ax.legend(
    title="Mechanism",
    ncol=5,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.20),
    frameon=False,
    columnspacing=1.2,
    handlelength=1.2,
    handletextpad=0.35,
    borderaxespad=0.2
)

plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight")
print(f"Saved figure to: {output_path}")

plt.show()