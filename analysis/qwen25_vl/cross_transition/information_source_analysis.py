#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re

import pandas as pd

INPUT_FILES = [
    "analysis/transition_groups/SAFE_to_SAFE.csv",
    "analysis/transition_groups/SAFE_to_UNSAFE.csv",
    "analysis/transition_groups/UNSAFE_to_SAFE.csv",
    "analysis/transition_groups/UNSAFE_to_UNSAFE.csv",
]

OUT_DIR = Path("analysis/cross_transition")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LONG = OUT_DIR / "isap_long.csv"
OUT_VARIANT_SUMMARY = OUT_DIR / "isap_variant_summary.csv"
OUT_SOURCE_SUMMARY = OUT_DIR / "isap_source_summary.csv"
OUT_MECH_VARIANT_PIVOT = OUT_DIR / "isap_mechanism_by_variant.csv"
OUT_MECH_TRANSITION_PIVOT = OUT_DIR / "isap_mechanism_by_transition.csv"
OUT_MECH_PROMPT_PIVOT = OUT_DIR / "isap_mechanism_by_prompt.csv"
OUT_SOURCE_MECH_PIVOT = OUT_DIR / "isap_source_by_mechanism.csv"
OUT_DOMINANT = OUT_DIR / "isap_dominant_source_candidates.csv"
OUT_MD = OUT_DIR / "isap_summary.md"

GROUP_ORDER = [
    "SAFE_to_SAFE",
    "SAFE_to_UNSAFE",
    "UNSAFE_to_SAFE",
    "UNSAFE_to_UNSAFE",
]

VARIANT_TO_MODALITY = {
    "SD": "Image",
    "SD_TYPO": "Image+Text",
    "TYPO": "Text",
}

VARIANT_TO_AVAILABLE_INFO = {
    "SD": "Visual scene + Rephrased Question(SD)",
    "SD_TYPO": "Visual scene + Embedded OCR text + Rephrased Question",
    "TYPO": "Embedded OCR text + Rephrased Question",
}

VARIANT_TO_SOURCE_SIGNATURE = {
    "SD": "Visual scene + SD prompt",
    "SD_TYPO": "Visual scene + OCR text + prompt",
    "TYPO": "OCR text + prompt",
}

GROUP_ALIASES = {
    "SAFE→SAFE": "SAFE_to_SAFE",
    "SAFE->SAFE": "SAFE_to_SAFE",
    "S→S": "SAFE_to_SAFE",
    "S->S": "SAFE_to_SAFE",
    "SAFE_to_SAFE": "SAFE_to_SAFE",
    "SAFE→UNSAFE": "SAFE_to_UNSAFE",
    "SAFE->UNSAFE": "SAFE_to_UNSAFE",
    "S→U": "SAFE_to_UNSAFE",
    "S->U": "SAFE_to_UNSAFE",
    "SAFE_to_UNSAFE": "SAFE_to_UNSAFE",
    "UNSAFE→SAFE": "UNSAFE_to_SAFE",
    "UNSAFE->SAFE": "UNSAFE_to_SAFE",
    "U→S": "UNSAFE_to_SAFE",
    "U->S": "UNSAFE_to_SAFE",
    "UNSAFE_to_SAFE": "UNSAFE_to_SAFE",
    "UNSAFE→UNSAFE": "UNSAFE_to_UNSAFE",
    "UNSAFE->UNSAFE": "UNSAFE_to_UNSAFE",
    "U→U": "UNSAFE_to_UNSAFE",
    "U->U": "UNSAFE_to_UNSAFE",
    "UNSAFE_to_UNSAFE": "UNSAFE_to_UNSAFE",
}


def normalize_group(value: object) -> str:
    text = str(value).strip()
    return GROUP_ALIASES.get(text, text)


def normalize_variant(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if upper in VARIANT_TO_MODALITY:
        return upper
    if "SD_TYPO" in upper:
        return "SD_TYPO"
    if upper == "SD":
        return "SD"
    if upper == "TYPO":
        return "TYPO"
    return text


def get_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find a {label} column. Tried: {candidates}. Found: {list(df.columns)}"
    )


def normalize_mechanism_name(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"^[\-\*\u2022]+\s*", "", text)
    text = re.sub(r"^\d+[\).\s]+", "", text)
    text = text.strip(" \t\n\r.;:,")
    text = re.sub(r"\s+", " ", text)
    return text


def split_mechanism_cell(value: object) -> List[str]:
    """
    Expand a behavioral_mechanism cell into individual mechanism names.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    raw_parts = re.split(r"[;\n]+", text.replace("\r", "\n"))

    mechanisms: List[str] = []
    for raw in raw_parts:
        part = raw.strip()
        if not part:
            continue

        # If a label is followed by explanatory text, keep only the label.
        if re.search(r"\s[–—]\s", part):
            part = re.split(r"\s[–—]\s", part, maxsplit=1)[0].strip()

        part = normalize_mechanism_name(part)
        if part and part not in mechanisms:
            mechanisms.append(part)

    return mechanisms


def confidence_bucket(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


def infer_prompt_candidate(sd: int, sd_typo: int, typo: int) -> Tuple[str, str, float]:
    """
    Infer whether the SD-specific prompt wording or the generic rephrased
    question appears more associated with the mechanism distribution.

    This is a descriptive heuristic, not a causal claim.
    """
    total = sd + sd_typo + typo
    sd_prompt_count = sd
    rephrased_question_count = sd_typo + typo

    if total == 0:
        return "Mixed / uncertain", "No occurrences across variants", 0.30

    if sd_prompt_count == 0:
        return (
            "Rephrased Question likely influential",
            f"No SD-prompt occurrences; all instances are under Rephrased Question ({rephrased_question_count}/{total})",
            0.70,
        )

    if rephrased_question_count == 0:
        return (
            "Rephrased Question(SD) likely influential",
            f"No generic Rephrased Question occurrences; all instances are under Rephrased Question(SD) ({sd_prompt_count}/{total})",
            0.70,
        )

    prompt_diff = abs(sd_prompt_count - rephrased_question_count)
    prompt_ratio = max(sd_prompt_count, rephrased_question_count) / total

    if prompt_diff <= max(2, int(0.15 * total)):
        return (
            "Prompt wording likely shared influence",
            f"Prompt variants are relatively balanced: SD prompt={sd_prompt_count}, Rephrased Question={rephrased_question_count}",
            0.55,
        )

    if sd_prompt_count > rephrased_question_count:
        return (
            "Rephrased Question(SD) likely influential",
            f"SD prompt dominates: {sd_prompt_count}/{total}",
            min(0.85, max(0.60, prompt_ratio)),
        )

    return (
        "Rephrased Question likely influential",
        f"Generic Rephrased Question dominates: {rephrased_question_count}/{total}",
        min(0.85, max(0.60, prompt_ratio)),
    )


def infer_source_candidate(sd: int, sd_typo: int, typo: int) -> Tuple[str, str, float]:
    """
    Infer the most plausible information source from the distribution
    across SD, SD_TYPO, and TYPO.

    The inference is intentionally conservative and distribution-based.
    """
    total = sd + sd_typo + typo
    if total == 0:
        return "Mixed / uncertain", "No occurrences across variants", 0.30

    image_bearing = sd + sd_typo
    ocr_bearing = sd_typo + typo

    counts = {"SD": sd, "SD_TYPO": sd_typo, "TYPO": typo}
    active = [k for k, v in counts.items() if v > 0]
    dominant_variant = max(counts, key=counts.get)
    dominant_variant_count = counts[dominant_variant]
    dominant_variant_ratio = dominant_variant_count / total

    # Strong single-variant support.
    if active == ["SD"]:
        score = 0.35 if total <= 2 else 0.55 if total <= 4 else 0.80
        return (
            "Visual scene + SD prompt",
            f"Observed only in SD ({sd}/{total})",
            score,
        )

    if active == ["SD_TYPO"]:
        score = 0.35 if total <= 2 else 0.55 if total <= 4 else 0.80
        return (
            "Visual scene + Embedded OCR text + Rephrased Question",
            f"Observed only in SD_TYPO ({sd_typo}/{total})",
            score,
        )

    if active == ["TYPO"]:
        score = 0.35 if total <= 2 else 0.55 if total <= 4 else 0.80
        return (
            "Embedded OCR text + Rephrased Question",
            f"Observed only in TYPO ({typo}/{total})",
            score,
        )

    # Strong family dominance.
    if ocr_bearing >= 0.70 * total and sd <= 0.15 * total:
        score = 0.85 if total >= 10 else 0.70
        return (
            "Embedded OCR text likely influential",
            f"OCR-bearing variants dominate ({ocr_bearing}/{total}); SD is weak ({sd}/{total})",
            score,
        )

    if image_bearing >= 0.70 * total and typo <= 0.15 * total:
        score = 0.85 if total >= 10 else 0.70
        return (
            "Visual scene likely influential",
            f"Image-bearing variants dominate ({image_bearing}/{total}); TYPO is weak ({typo}/{total})",
            score,
        )

    # Concentration in a single variant, but not enough to identify a clear source family.
    if dominant_variant_ratio >= 0.55:
        if dominant_variant == "SD":
            return (
                "Visual scene likely influential",
                f"Concentrated in SD ({sd}/{total})",
                0.65,
            )
        if dominant_variant == "SD_TYPO":
            return (
                "Visual scene + Embedded OCR text likely influential",
                f"Concentrated in SD_TYPO ({sd_typo}/{total})",
                0.65,
            )
        return (
            "Embedded OCR text likely influential",
            f"Concentrated in TYPO ({typo}/{total})",
            0.65,
        )

    # Rough balance across all variants.
    spread = max(counts.values()) - min(counts.values())
    if spread <= max(2, int(0.15 * total)):
        return (
            "Multisource / robust across variants",
            f"Counts are relatively balanced across SD={sd}, SD_TYPO={sd_typo}, TYPO={typo}",
            0.60,
        )

    # Fall back to prompt/shared instruction influence when no single source family dominates.
    return (
        "Shared instruction / prompt influence likely",
        f"No single source family clearly dominates: SD={sd}, SD_TYPO={sd_typo}, TYPO={typo}",
        0.50,
    )


def main() -> None:
    frames = []

    for file_path in INPUT_FILES:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)

        variant_col = get_column(df, ["variant", "Variant"], "variant")
        mech_col = get_column(df, ["behavioral_mechanism", "Behavioral Mechanism"], "behavioral mechanism")
        trans_col = get_column(df, ["transition", "Transition"], "transition")

        optional_context_cols = []
        for c in [
            "question_id",
            "original_question",
            "changed_question",
            "key_phrase",
            "phrase_type",
            "prompt",
            "original_label",
            "masked_label",
        ]:
            if c in df.columns:
                optional_context_cols.append(c)

        temp_cols = optional_context_cols + [variant_col, mech_col, trans_col]
        temp = df[temp_cols].copy()

        rename_map = {
            variant_col: "variant",
            mech_col: "Behavioral Mechanism",
            trans_col: "transition",
        }
        temp = temp.rename(columns=rename_map)

        temp["transition_group"] = normalize_group(path.stem.replace(".csv", ""))
        temp["variant"] = temp["variant"].map(normalize_variant)
        temp["input_modality"] = temp["variant"].map(VARIANT_TO_MODALITY)
        temp["available_information"] = temp["variant"].map(VARIANT_TO_AVAILABLE_INFO)
        temp["source_signature"] = temp["variant"].map(VARIANT_TO_SOURCE_SIGNATURE)
        temp["prompt_variant"] = temp["variant"].map(
            lambda v: "Rephrased Question(SD)" if v == "SD" else "Rephrased Question"
        )

        # Expand multi-mechanism cells into one row per mechanism.
        temp["Behavioral Mechanism"] = temp["Behavioral Mechanism"].apply(split_mechanism_cell)
        temp = temp.explode("Behavioral Mechanism", ignore_index=True)
        temp["Behavioral Mechanism"] = temp["Behavioral Mechanism"].apply(normalize_mechanism_name)
        temp = temp[temp["Behavioral Mechanism"] != ""].copy()

        frames.append(temp)

    if not frames:
        raise ValueError("No input files were loaded.")

    long_df = pd.concat(frames, ignore_index=True)

    long_df["transition_group"] = pd.Categorical(
        long_df["transition_group"], categories=GROUP_ORDER, ordered=True
    )
    long_df["variant"] = pd.Categorical(
        long_df["variant"], categories=["SD", "SD_TYPO", "TYPO"], ordered=True
    )
    long_df = long_df.sort_values(
        ["variant", "transition_group", "Behavioral Mechanism"]
    ).reset_index(drop=True)

    long_df.to_csv(OUT_LONG, index=False)

    # Variant-level summary
    variant_summary = (
        long_df.groupby(["variant", "transition_group"], as_index=False, observed=False)
        .size()
        .rename(columns={"size": "sample_count"})
        .sort_values(["variant", "transition_group"])
    )
    variant_summary.to_csv(OUT_VARIANT_SUMMARY, index=False)

    # Source-level summary
    source_summary = (
        long_df.groupby(["input_modality", "transition_group"], as_index=False, observed=False)
        .size()
        .rename(columns={"size": "sample_count"})
        .sort_values(["input_modality", "transition_group"])
    )
    source_summary.to_csv(OUT_SOURCE_SUMMARY, index=False)

    # Mechanism x variant pivot
    mech_variant = (
        long_df.pivot_table(
            index="Behavioral Mechanism",
            columns="variant",
            values="transition_group",
            aggfunc="count",
            fill_value=0,
            observed=False,
        )
        .reindex(columns=["SD", "SD_TYPO", "TYPO"], fill_value=0)
        .reset_index()
    )
    mech_variant.to_csv(OUT_MECH_VARIANT_PIVOT, index=False)

    # Mechanism x transition pivot
    mech_transition = (
        long_df.pivot_table(
            index="Behavioral Mechanism",
            columns="transition_group",
            values="variant",
            aggfunc="count",
            fill_value=0,
            observed=False,
        )
        .reindex(columns=GROUP_ORDER, fill_value=0)
        .reset_index()
    )
    mech_transition.to_csv(OUT_MECH_TRANSITION_PIVOT, index=False)

    # Mechanism x prompt pivot
    mech_prompt = (
        long_df.pivot_table(
            index="Behavioral Mechanism",
            columns="prompt_variant",
            values="variant",
            aggfunc="count",
            fill_value=0,
            observed=False,
        )
        .reset_index()
    )
    mech_prompt.to_csv(OUT_MECH_PROMPT_PIVOT, index=False)

    # Source signature x mechanism pivot
    source_mech = (
        long_df.pivot_table(
            index="Behavioral Mechanism",
            columns="source_signature",
            values="variant",
            aggfunc="count",
            fill_value=0,
            observed=False,
        )
        .reset_index()
    )
    source_mech.to_csv(OUT_SOURCE_MECH_PIVOT, index=False)

    # Dominant source candidates by mechanism
    dom_rows = []
    for mech, sub in long_df.groupby("Behavioral Mechanism", observed=False):
        sd = int((sub["variant"] == "SD").sum())
        sd_typo = int((sub["variant"] == "SD_TYPO").sum())
        typo = int((sub["variant"] == "TYPO").sum())
        total_occurrences = sd + sd_typo + typo

        variants_present = [v for v, c in [("SD", sd), ("SD_TYPO", sd_typo), ("TYPO", typo)] if c > 0]
        dominant_variant = max([("SD", sd), ("SD_TYPO", sd_typo), ("TYPO", typo)], key=lambda x: x[1])[0]
        dominant_variant_count = max(sd, sd_typo, typo)
        dominance_ratio = (dominant_variant_count / total_occurrences) if total_occurrences else 0.0

        source_candidate, source_rationale, source_conf = infer_source_candidate(sd, sd_typo, typo)
        source_conf_level = confidence_bucket(source_conf)

        prompt_candidate, prompt_rationale, prompt_conf = infer_prompt_candidate(sd, sd_typo, typo)
        prompt_conf_level = confidence_bucket(prompt_conf)

        dom_rows.append(
            {
                "Behavioral Mechanism": mech,
                "SD_count": sd,
                "SD_TYPO_count": sd_typo,
                "TYPO_count": typo,
                "image_bearing_count": sd + sd_typo,
                "ocr_bearing_count": sd_typo + typo,
                "image_bearing_ratio": round((sd + sd_typo) / total_occurrences, 4) if total_occurrences else 0.0,
                "ocr_bearing_ratio": round((sd_typo + typo) / total_occurrences, 4) if total_occurrences else 0.0,
                "SD_prompt_count": sd,
                "Rephrased_Question_count": sd_typo + typo,
                "prompt_dominance_ratio": round(max(sd, sd_typo + typo) / total_occurrences, 4) if total_occurrences else 0.0,
                "variants_present": ", ".join(variants_present),
                "variant_support_count": len(variants_present),
                "dominant_variant": dominant_variant,
                "dominant_variant_count": dominant_variant_count,
                "dominance_ratio": round(dominance_ratio, 4),
                "source_candidate": source_candidate,
                "source_confidence_score": round(source_conf, 4),
                "source_confidence_level": source_conf_level,
                "source_rationale": source_rationale,
                "prompt_candidate": prompt_candidate,
                "prompt_confidence_score": round(prompt_conf, 4),
                "prompt_confidence_level": prompt_conf_level,
                "prompt_rationale": prompt_rationale,
            }
        )

    dominant_df = pd.DataFrame(dom_rows).sort_values(
        ["variant_support_count", "dominant_variant_count", "Behavioral Mechanism"],
        ascending=[False, False, True],
    )
    dominant_df.to_csv(OUT_DOMINANT, index=False)

    # Markdown report
    md: List[str] = []
    md.append("# Information Source Analysis Protocol (ISAP)\n\n")

    md.append("## Variant summary\n\n")
    md.append(variant_summary.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Source summary\n\n")
    md.append(source_summary.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Mechanism by variant\n\n")
    md.append(mech_variant.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Mechanism by transition group\n\n")
    md.append(mech_transition.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Mechanism by prompt variant\n\n")
    md.append(mech_prompt.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Source signature by mechanism\n\n")
    md.append(source_mech.to_markdown(index=False))
    md.append("\n\n")

    md.append("## Dominant source candidates\n\n")
    md.append(dominant_df.to_markdown(index=False))
    md.append("\n")

    OUT_MD.write_text("".join(md), encoding="utf-8")

    print(f"Saved long table -> {OUT_LONG}")
    print(f"Saved variant summary -> {OUT_VARIANT_SUMMARY}")
    print(f"Saved source summary -> {OUT_SOURCE_SUMMARY}")
    print(f"Saved mechanism-by-variant pivot -> {OUT_MECH_VARIANT_PIVOT}")
    print(f"Saved mechanism-by-transition pivot -> {OUT_MECH_TRANSITION_PIVOT}")
    print(f"Saved mechanism-by-prompt pivot -> {OUT_MECH_PROMPT_PIVOT}")
    print(f"Saved source-by-mechanism pivot -> {OUT_SOURCE_MECH_PIVOT}")
    print(f"Saved dominant source candidates -> {OUT_DOMINANT}")
    print(f"Saved markdown report -> {OUT_MD}")


if __name__ == "__main__":
    main()