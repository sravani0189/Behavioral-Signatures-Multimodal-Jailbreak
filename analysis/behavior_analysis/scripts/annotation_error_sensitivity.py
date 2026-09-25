#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


# ============================================================
# Paths
# ============================================================

ROOT = Path("analysis/behavior_analysis")

INPUT = ROOT / "outputs" / "fewshot" / "predictions_complete_clean.csv"

OUT_REPLICATES = (
    ROOT / "outputs" / "fewshot" /
    "annotation_error_sensitivity_replicates.csv"
)

OUT_SUMMARY = (
    ROOT / "outputs" / "fewshot" /
    "annotation_error_sensitivity_summary.csv"
)


# ============================================================
# Configuration
# ============================================================

N_REPLICATES = 1000
RANDOM_SEED = 42

# Error rates implied by the 50-sample GPT-4o-mini pilot:
# Effect accuracy = 82%       -> error rate = 18%
# Attribution accuracy = 72%  -> error rate = 28%
EFFECT_ERROR_RATE = 0.18
ATTRIBUTION_ERROR_RATE = 0.28

N_CLUSTERS = 3


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT)

required = {
    "model",
    "behavioral_effect_pred",
    "primary_driver_pred",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df.dropna(
    subset=[
        "model",
        "behavioral_effect_pred",
        "primary_driver_pred",
    ]
).copy()

df["model"] = df["model"].astype(str)
df["behavioral_effect_pred"] = df["behavioral_effect_pred"].astype(str)
df["primary_driver_pred"] = df["primary_driver_pred"].astype(str)


# Freeze observed category spaces.
models = sorted(df["model"].unique())

effect_labels = sorted(
    df["behavioral_effect_pred"].unique()
)

attribution_labels = sorted(
    df["primary_driver_pred"].unique()
)

print("=" * 78)
print("ANNOTATION-ERROR SENSITIVITY ANALYSIS")
print("=" * 78)
print(f"Rows: {len(df)}")
print(f"Models: {len(models)}")
print(f"Observed Effect categories: {len(effect_labels)}")
print(f"Observed Attribution categories: {len(attribution_labels)}")
print(f"Effect perturbation rate: {EFFECT_ERROR_RATE:.2f}")
print(f"Attribution perturbation rate: {ATTRIBUTION_ERROR_RATE:.2f}")
print(f"Replicates: {N_REPLICATES}")


# ============================================================
# Signature construction
# ============================================================

def make_signature(data):
    """
    Construct transition-excluded Effect--Attribution signatures.

    Each component is a within-model percentage distribution.
    """
    rows = []

    for model in models:
        g = data[data["model"] == model]

        effect_counts = (
            g["behavioral_effect_pred"]
            .value_counts(normalize=True)
            .reindex(effect_labels, fill_value=0.0)
            * 100.0
        )

        attr_counts = (
            g["primary_driver_pred"]
            .value_counts(normalize=True)
            .reindex(attribution_labels, fill_value=0.0)
            * 100.0
        )

        # Match BS^{EA}: Effect followed by Attribution.
        vector = np.concatenate(
            [
                effect_counts.to_numpy(dtype=float),
                attr_counts.to_numpy(dtype=float),
            ]
        )

        rows.append(vector)

    return np.vstack(rows)


def cluster_signature(X):
    clustering = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        linkage="ward",
    )
    return clustering.fit_predict(X)


# ============================================================
# Observed partition
# ============================================================

X_observed = make_signature(df)
labels_observed = cluster_signature(X_observed)

print("\nObserved partition:")
for model, label in zip(models, labels_observed):
    print(f"  {model:22s} -> cluster {label}")


# ============================================================
# Label perturbation
# ============================================================

def perturb_column(values, label_space, error_rate, rng):
    """
    Independently perturb each label with probability error_rate.

    When perturbed, replace the observed label uniformly with one of
    the OTHER observed labels in that taxonomy.
    """
    values = np.asarray(values, dtype=object).copy()

    change_mask = rng.random(len(values)) < error_rate
    changed_indices = np.flatnonzero(change_mask)

    for idx in changed_indices:
        current = values[idx]

        alternatives = [
            label for label in label_space
            if label != current
        ]

        if alternatives:
            values[idx] = rng.choice(alternatives)

    return values, len(changed_indices)


# ============================================================
# Monte Carlo sensitivity
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

records = []

for replicate in range(1, N_REPLICATES + 1):

    perturbed = df.copy()

    new_effects, n_effect_changed = perturb_column(
        perturbed["behavioral_effect_pred"].to_numpy(),
        effect_labels,
        EFFECT_ERROR_RATE,
        rng,
    )

    new_attrs, n_attr_changed = perturb_column(
        perturbed["primary_driver_pred"].to_numpy(),
        attribution_labels,
        ATTRIBUTION_ERROR_RATE,
        rng,
    )

    perturbed["behavioral_effect_pred"] = new_effects
    perturbed["primary_driver_pred"] = new_attrs

    X_perturbed = make_signature(perturbed)
    labels_perturbed = cluster_signature(X_perturbed)

    ari = adjusted_rand_score(
        labels_observed,
        labels_perturbed,
    )

    records.append(
        {
            "replicate": replicate,
            "ARI": ari,
            "effect_labels_changed": n_effect_changed,
            "attribution_labels_changed": n_attr_changed,
            "exact_partition_recovery": int(
                np.isclose(ari, 1.0)
            ),
        }
    )


results = pd.DataFrame(records)

results.to_csv(
    OUT_REPLICATES,
    index=False,
)


# ============================================================
# Summary
# ============================================================

summary = pd.DataFrame(
    [
        {
            "replicates": N_REPLICATES,
            "effect_error_rate": EFFECT_ERROR_RATE,
            "attribution_error_rate": ATTRIBUTION_ERROR_RATE,
            "median_ARI": results["ARI"].median(),
            "mean_ARI": results["ARI"].mean(),
            "ARI_2.5_percentile": results["ARI"].quantile(0.025),
            "ARI_97.5_percentile": results["ARI"].quantile(0.975),
            "exact_recovery_n": results[
                "exact_partition_recovery"
            ].sum(),
            "exact_recovery_pct": (
                100
                * results["exact_partition_recovery"].mean()
            ),
            "mean_effect_labels_changed": results[
                "effect_labels_changed"
            ].mean(),
            "mean_attribution_labels_changed": results[
                "attribution_labels_changed"
            ].mean(),
        }
    ]
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


print("\n" + "=" * 78)
print("RESULTS")
print("=" * 78)

print(f"Median ARI: {results['ARI'].median():.4f}")
print(f"Mean ARI:   {results['ARI'].mean():.4f}")
print(
    "2.5--97.5 percentile: "
    f"{results['ARI'].quantile(0.025):.4f} -- "
    f"{results['ARI'].quantile(0.975):.4f}"
)

exact_n = results["exact_partition_recovery"].sum()
exact_pct = 100 * results["exact_partition_recovery"].mean()

print(f"Exact recovery: {exact_n}/{N_REPLICATES} ({exact_pct:.1f}%)")

print(
    "Mean Effect labels perturbed: "
    f"{results['effect_labels_changed'].mean():.1f}"
)

print(
    "Mean Attribution labels perturbed: "
    f"{results['attribution_labels_changed'].mean():.1f}"
)

print("\nSaved:")
print(OUT_REPLICATES)
print(OUT_SUMMARY)
