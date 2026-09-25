import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def main():
    # Load pairwise data
    csv_path = Path("analysis/behavior_analysis/outputs/fewshot/pairwise_signature_attack_table.csv")
    df = pd.read_csv(csv_path)

    # Color points: same family vs different family
    colors = df["SameFamily"].map({1: "blue", 0: "red"})

    plt.figure(figsize=(9, 6.5))

    plt.scatter(
        df["AttackDiff_Masked"],
        df["SigDistance"],
        c=colors,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.5,
        s=45
    )

    # Threshold line
    plt.axvline(
        x=0.05,
        linestyle="--",
        color="black",
        linewidth=1.5
    )

    # Axis labels and title
    plt.xlabel("Attack Success Rate Difference", fontsize=15)
    plt.ylabel("Behavioral Distance", fontsize=15)
    plt.title("Behavioral Distance vs Attack Success Rate Difference", fontsize=16)

    # Tick labels
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # Legend
    blue_patch = mpatches.Patch(color="blue", label="Same Behavioral Family")
    red_patch = mpatches.Patch(color="red", label="Different Behavioral Family")

    plt.legend(
        handles=[blue_patch, red_patch],
        fontsize=12,
        loc="lower right"
    )

    # Grid
    plt.grid(alpha=0.3)

    # Save figure
    plt.tight_layout()
    plt.savefig(
        "figure6_asr_vs_behavior_clean.pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()