#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

import math
import pandas as pd

# Try to use scipy for p-values; if unavailable, raise a clear error.
try:
    from scipy.stats import chi2 as chi2_dist
except Exception as e:  # pragma: no cover
    chi2_dist = None
    _SCIPY_IMPORT_ERROR = e


# -----------------------------
# Paths
# -----------------------------
INPUT_VARIANT_CSV = Path(
    "analysis/behavior_analysis/outputs/isap_full_models/All_Models/isap_mechanism_by_variant.csv"
)
INPUT_DOMINANT_CSV = Path(
    "analysis/behavior_analysis/outputs/isap_full_models/All_Models/isap_dominant_source_candidates.csv"
)

OUTPUT_DIR = Path("analysis/behavior_analysis/outputs/isap_statistics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "isap_statistics.csv"
OUTPUT_MD = OUTPUT_DIR / "isap_statistics.md"


# -----------------------------
# Configuration
# -----------------------------
VARIANT_COLS = ["SD", "SD_TYPO", "TYPO"]
MIN_EXPECTED_COUNT = 5.0


# -----------------------------
# Helpers
# -----------------------------
def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")


def cramers_v_gof(chi2: float, n: int, k: int) -> float:
    """
    Cramér's V for a goodness-of-fit test with k categories.
    For a one-dimensional goodness-of-fit setting, we use:
        V = sqrt(chi2 / (N * (k - 1)))
    """
    if n <= 0 or k <= 1:
        return 0.0
    return math.sqrt(max(0.0, chi2) / (n * (k - 1)))


def chi_square_gof(observed: list[int]) -> dict[str, float]:
    """
    Chi-square goodness-of-fit test against a uniform distribution across variants.
    Returns chi2, df, p_value, expected count, min expected.
    """
    if chi2_dist is None:  # pragma: no cover
        raise ImportError(
            "scipy is required for p-values in run_isap_statistics.py. "
            f"Import error: {_SCIPY_IMPORT_ERROR}"
        )

    n = sum(observed)
    k = len(observed)
    if n == 0 or k <= 1:
        return {
            "chi2": 0.0,
            "df": max(0, k - 1),
            "p_value": 1.0,
            "expected": 0.0,
            "min_expected": 0.0,
        }

    expected = n / k
    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    df = k - 1
    p_value = float(chi2_dist.sf(chi2, df))

    return {
        "chi2": float(chi2),
        "df": int(df),
        "p_value": p_value,
        "expected": float(expected),
        "min_expected": float(expected),  # uniform in this setup
    }


def infer_source_from_variant_bias(sd: int, sd_typo: int, typo: int) -> tuple[str, str]:
    """
    Simple descriptive mapping from the observed variant distribution to an
    inferred information source. This mirrors the ISAP heuristic framing.
    """
    total = sd + sd_typo + typo
    if total == 0:
        return "Mixed / uncertain", "No occurrences across variants"

    image_bearing = sd + sd_typo
    ocr_bearing = sd_typo + typo

    # Strong OCR-bearing concentration
    if ocr_bearing >= 0.70 * total and sd <= 0.15 * total:
        return "Embedded OCR text likely influential", (
            f"OCR-bearing variants dominate ({ocr_bearing}/{total}); SD is weak ({sd}/{total})"
        )

    # Strong image-bearing concentration
    if image_bearing >= 0.70 * total and typo <= 0.15 * total:
        return "Visual scene likely influential", (
            f"Image-bearing variants dominate ({image_bearing}/{total}); TYPO is weak ({typo}/{total})"
        )

    # Rough balance across all three
    spread = max(sd, sd_typo, typo) - min(sd, sd_typo, typo)
    if spread <= max(2, int(0.15 * total)):
        return "Multisource / robust across variants", (
            f"Counts are relatively balanced across SD={sd}, SD_TYPO={sd_typo}, TYPO={typo}"
        )

    # Fallback
    return "Shared instruction / prompt influence likely", (
        f"No single source family clearly dominates: SD={sd}, SD_TYPO={sd_typo}, TYPO={typo}"
    )


def confidence_from_p_value(p_value: float, effect: float, low_expected: bool) -> str:
    """
    A simple confidence label for the paper:
      - High: strong significance, non-trivial effect, and adequate expected counts
      - Medium: some support but not all criteria met
      - Low: weak evidence or low expected counts
    """
    if low_expected:
        return "Low"
    if p_value < 0.001 and effect >= 0.30:
        return "High"
    if p_value < 0.05 and effect >= 0.15:
        return "Medium"
    return "Low"


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    require(INPUT_VARIANT_CSV)
    require(INPUT_DOMINANT_CSV)

    variant_df = pd.read_csv(INPUT_VARIANT_CSV)
    dominant_df = pd.read_csv(INPUT_DOMINANT_CSV)

    # Make sure the variant table has the expected columns
    missing = [c for c in VARIANT_COLS if c not in variant_df.columns]
    if missing:
        raise KeyError(
            f"INPUT_VARIANT_CSV is missing expected columns {missing}. "
            f"Found columns: {list(variant_df.columns)}"
        )

    # Normalize mechanism names in both tables
    variant_df["Behavioral Mechanism"] = variant_df["Behavioral Mechanism"].astype(str).str.strip()
    dominant_df["Behavioral Mechanism"] = dominant_df["Behavioral Mechanism"].astype(str).str.strip()

    # Ensure numeric counts
    for c in VARIANT_COLS:
        variant_df[c] = pd.to_numeric(variant_df[c], errors="coerce").fillna(0).astype(int)

    # Merge in the heuristic source candidate from the dominant-source table
    dominant_cols = [
        "Behavioral Mechanism",
        "source_candidate",
        "source_confidence_score",
        "source_confidence_level",
        "source_rationale",
        "prompt_candidate",
        "prompt_confidence_score",
        "prompt_confidence_level",
        "prompt_rationale",
    ]
    dominant_keep = [c for c in dominant_cols if c in dominant_df.columns]
    dom_small = dominant_df[dominant_keep].copy()

    merged = variant_df.merge(dom_small, on="Behavioral Mechanism", how="left")

    rows = []
    for _, row in merged.iterrows():
        mech = row["Behavioral Mechanism"]
        sd = int(row["SD"])
        sd_typo = int(row["SD_TYPO"])
        typo = int(row["TYPO"])
        observed = [sd, sd_typo, typo]
        total = sum(observed)

        stats = chi_square_gof(observed)
        chi2 = stats["chi2"]
        df = stats["df"]
        p_value = stats["p_value"]

        expected = total / 3 if total else 0.0
        low_expected = expected < MIN_EXPECTED_COUNT

        # Effect size
        v = cramers_v_gof(chi2=chi2, n=total, k=3)

        # Descriptive inference, matching your ISAP logic
        inferred_source, inferred_rationale = infer_source_from_variant_bias(sd, sd_typo, typo)
        confidence = confidence_from_p_value(p_value, v, low_expected)

        rows.append(
            {
                "Behavioral Mechanism": mech,
                "SD": sd,
                "SD_TYPO": sd_typo,
                "TYPO": typo,
                "Total": total,
                "Expected_Per_Variant": round(expected, 4),
                "Chi2": round(chi2, 4),
                "DF": df,
                "p_value": p_value,
                "Cramers_V": round(v, 4),
                "Low_Expected_Count_Flag": bool(low_expected),
                "Significant_0.05": bool(p_value < 0.05),
                "Inferred_Source": inferred_source,
                "Inference_Rationale": inferred_rationale,
                "Confidence": confidence,
                "source_candidate": row.get("source_candidate", ""),
                "source_confidence_score": row.get("source_confidence_score", ""),
                "source_confidence_level": row.get("source_confidence_level", ""),
                "prompt_candidate": row.get("prompt_candidate", ""),
                "prompt_confidence_score": row.get("prompt_confidence_score", ""),
                "prompt_confidence_level": row.get("prompt_confidence_level", ""),
            }
        )

    out_df = pd.DataFrame(rows)

    # Sort by significance / effect size / total
    out_df = out_df.sort_values(
        ["Significant_0.05", "Cramers_V", "Total", "Behavioral Mechanism"],
        ascending=[False, False, False, True],
    )

    out_df.to_csv(OUTPUT_CSV, index=False)

    # Markdown report
    md = []
    md.append("# ISAP Statistical Validation\n\n")
    md.append(
        "This report tests whether each behavioral mechanism is uniformly distributed "
        "across SD, SD_TYPO, and TYPO using a chi-square goodness-of-fit test. "
        "Mechanisms with low expected counts are flagged as less reliable for inference.\n\n"
    )

    md.append("## Summary table\n\n")
    md.append(out_df.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Notes\n\n")
    md.append(
        f"- Significance threshold: 0.05\n"
        f"- Low expected count threshold: expected count < {MIN_EXPECTED_COUNT}\n"
        f"- Cramér's V is used as an effect size summary\n"
    )

    OUTPUT_MD.write_text("".join(md), encoding="utf-8")

    print(f"Saved statistical summary -> {OUTPUT_CSV}")
    print(f"Saved markdown report -> {OUTPUT_MD}")


if __name__ == "__main__":
    main()