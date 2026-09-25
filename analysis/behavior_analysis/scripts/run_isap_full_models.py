#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re

import pandas as pd

# -----------------------------
# Input / output configuration
# -----------------------------
INPUT_CSV = Path("analysis/behavior_analysis/outputs/fewshot/predictions_complete_clean.csv")
OUTPUT_ROOT = Path("analysis/behavior_analysis/outputs/isap_full_models")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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
    "SAFE -> SAFE": "SAFE_to_SAFE",
    "S→S": "SAFE_to_SAFE",
    "S->S": "SAFE_to_SAFE",
    "SAFE_to_SAFE": "SAFE_to_SAFE",

    "SAFE→UNSAFE": "SAFE_to_UNSAFE",
    "SAFE->UNSAFE": "SAFE_to_UNSAFE",
    "SAFE -> UNSAFE": "SAFE_to_UNSAFE",
    "S→U": "SAFE_to_UNSAFE",
    "S->U": "SAFE_to_UNSAFE",
    "SAFE_to_UNSAFE": "SAFE_to_UNSAFE",

    "UNSAFE→SAFE": "UNSAFE_to_SAFE",
    "UNSAFE->SAFE": "UNSAFE_to_SAFE",
    "UNSAFE -> SAFE": "UNSAFE_to_SAFE",
    "U→S": "UNSAFE_to_SAFE",
    "U->S": "UNSAFE_to_SAFE",
    "UNSAFE_to_SAFE": "UNSAFE_to_SAFE",

    "UNSAFE→UNSAFE": "UNSAFE_to_UNSAFE",
    "UNSAFE->UNSAFE": "UNSAFE_to_UNSAFE",
    "UNSAFE -> UNSAFE": "UNSAFE_to_UNSAFE",
    "U→U": "UNSAFE_to_UNSAFE",
    "U->U": "UNSAFE_to_UNSAFE",
    "UNSAFE_to_UNSAFE": "UNSAFE_to_UNSAFE",
}


# -----------------------------
# Helpers
# -----------------------------
def sanitize_name(name: object) -> str:
    text = str(name).strip()
    text = re.sub(r"[^\w\s.-]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def get_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find a {label} column. Tried: {candidates}. Found: {list(df.columns)}"
    )


def normalize_group(value: object) -> str:
    text = str(value).strip()
    text = text.replace("→", "->")
    text = re.sub(r"\s*-\s*>\s*", "->", text)
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


def normalize_mechanism_name(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"^[\-\*\u2022]+\s*", "", text)
    text = re.sub(r"^\d+[\).\s]+", "", text)
    text = text.strip(" \t\n\r.;:,")
    text = re.sub(r"\s+", " ", text)
    return text


def split_mechanism_cell(value: object) -> List[str]:
    """
    Expand a behavioral mechanism cell into individual mechanism names.
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
    Infer the most plausible information source from the distribution across variants.
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

    spread = max(counts.values()) - min(counts.values())
    if spread <= max(2, int(0.15 * total)):
        return (
            "Multisource / robust across variants",
            f"Counts are relatively balanced across SD={sd}, SD_TYPO={sd_typo}, TYPO={typo}",
            0.60,
        )

    return (
        "Shared instruction / prompt influence likely",
        f"No single source family clearly dominates: SD={sd}, SD_TYPO={sd_typo}, TYPO={typo}",
        0.50,
    )


def build_model_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the prediction table into the schema expected by ISAP.
    """
    model_col = get_column(raw, ["model", "Model"], "model")
    variant_col = get_column(raw, ["variant", "Variant"], "variant")
    transition_col = get_column(raw, ["transition", "Transition"], "transition")
    mech_col = get_column(
        raw,
        ["primary_driver_pred", "behavioral_mechanism", "Behavioral Mechanism", "mechanism_pred"],
        "behavioral mechanism",
    )

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
        "model_family",
        "family",
    ]:
        if c in raw.columns:
            optional_context_cols.append(c)

    temp_cols = optional_context_cols + [model_col, variant_col, mech_col, transition_col]
    df = raw[temp_cols].copy()

    rename_map = {
        model_col: "model",
        variant_col: "variant",
        mech_col: "Behavioral Mechanism",
        transition_col: "transition",
    }
    df = df.rename(columns=rename_map)

    df["variant"] = df["variant"].map(normalize_variant)
    df["transition_group"] = df["transition"].map(normalize_group)
    df["input_modality"] = df["variant"].map(VARIANT_TO_MODALITY)
    df["available_information"] = df["variant"].map(VARIANT_TO_AVAILABLE_INFO)
    df["source_signature"] = df["variant"].map(VARIANT_TO_SOURCE_SIGNATURE)
    df["prompt_variant"] = df["variant"].map(
        lambda v: "Rephrased Question(SD)" if v == "SD" else "Rephrased Question"
    )

    df["Behavioral Mechanism"] = df["Behavioral Mechanism"].apply(split_mechanism_cell)
    df = df.explode("Behavioral Mechanism", ignore_index=True)
    df["Behavioral Mechanism"] = df["Behavioral Mechanism"].apply(normalize_mechanism_name)
    df = df[df["Behavioral Mechanism"] != ""].copy()

    df["transition_group"] = pd.Categorical(
        df["transition_group"], categories=GROUP_ORDER, ordered=True
    )
    df["variant"] = pd.Categorical(
        df["variant"], categories=["SD", "SD_TYPO", "TYPO"], ordered=True
    )

    return df


def run_isap(df: pd.DataFrame, out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    out_long = out_dir / "isap_long.csv"
    out_variant_summary = out_dir / "isap_variant_summary.csv"
    out_source_summary = out_dir / "isap_source_summary.csv"
    out_mech_variant = out_dir / "isap_mechanism_by_variant.csv"
    out_mech_transition = out_dir / "isap_mechanism_by_transition.csv"
    out_mech_prompt = out_dir / "isap_mechanism_by_prompt.csv"
    out_source_mech = out_dir / "isap_source_by_mechanism.csv"
    out_dominant = out_dir / "isap_dominant_source_candidates.csv"
    out_md = out_dir / "isap_summary.md"

    if df.empty:
        raise ValueError(f"No rows available for ISAP run: {label}")

    long_df = df.copy()

    # Sort for consistent outputs
    long_df = long_df.sort_values(
        ["variant", "transition_group", "Behavioral Mechanism"]
    ).reset_index(drop=True)
    long_df.to_csv(out_long, index=False)

    variant_summary = (
        long_df.groupby(["variant", "transition_group"], as_index=False, observed=False)
        .size()
        .rename(columns={"size": "sample_count"})
        .sort_values(["variant", "transition_group"])
    )
    variant_summary.to_csv(out_variant_summary, index=False)

    source_summary = (
        long_df.groupby(["input_modality", "transition_group"], as_index=False, observed=False)
        .size()
        .rename(columns={"size": "sample_count"})
        .sort_values(["input_modality", "transition_group"])
    )
    source_summary.to_csv(out_source_summary, index=False)

    mech_variant = (
        long_df.groupby(["Behavioral Mechanism", "variant"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["SD", "SD_TYPO", "TYPO"], fill_value=0)
        .reset_index()
    )
    mech_variant.to_csv(out_mech_variant, index=False)

    mech_transition = (
        long_df.groupby(["Behavioral Mechanism", "transition_group"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=GROUP_ORDER, fill_value=0)
        .reset_index()
    )
    mech_transition.to_csv(out_mech_transition, index=False)

    mech_prompt = (
        long_df.groupby(["Behavioral Mechanism", "prompt_variant"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    mech_prompt.to_csv(out_mech_prompt, index=False)

    source_mech = (
        long_df.groupby(["Behavioral Mechanism", "source_signature"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    source_mech.to_csv(out_source_mech, index=False)

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
    dominant_df.to_csv(out_dominant, index=False)

    # Markdown report
    md: List[str] = []
    md.append(f"# Information Source Analysis Protocol (ISAP): {label}\n\n")

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

    out_md.write_text("".join(md), encoding="utf-8")

    print(f"[{label}] Saved long table -> {out_long}")
    print(f"[{label}] Saved variant summary -> {out_variant_summary}")
    print(f"[{label}] Saved source summary -> {out_source_summary}")
    print(f"[{label}] Saved mechanism-by-variant pivot -> {out_mech_variant}")
    print(f"[{label}] Saved mechanism-by-transition pivot -> {out_mech_transition}")
    print(f"[{label}] Saved mechanism-by-prompt pivot -> {out_mech_prompt}")
    print(f"[{label}] Saved source-by-mechanism pivot -> {out_source_mech}")
    print(f"[{label}] Saved dominant source candidates -> {out_dominant}")
    print(f"[{label}] Saved markdown report -> {out_md}")


def main() -> None:
    require(INPUT_CSV)

    raw = pd.read_csv(INPUT_CSV)
    df = build_model_dataframe(raw)

    # Per-model outputs
    combined_frames = []
    for model_name, model_df in df.groupby("model", observed=False):
        model_label = str(model_name)
        model_out_dir = OUTPUT_ROOT / sanitize_name(model_label)
        run_isap(model_df.drop(columns=["model"]).copy(), model_out_dir, model_label)
        combined_frames.append(model_df)

    # Combined all-models output
    all_df = pd.concat(combined_frames, ignore_index=True)
    all_out_dir = OUTPUT_ROOT / "All_Models"
    run_isap(all_df.drop(columns=["model"]).copy(), all_out_dir, "All_Models")

    # Cross-model comparison table
    comparison_rows = []
    for model_name, model_df in df.groupby("model", observed=False):
        tmp = model_df.copy()
        mech_counts = tmp["Behavioral Mechanism"].value_counts()
        top_mech = mech_counts.index[0] if not mech_counts.empty else ""
        top_mech_count = int(mech_counts.iloc[0]) if not mech_counts.empty else 0

        variant_counts = tmp["variant"].value_counts().to_dict()
        transition_counts = tmp["transition_group"].value_counts().to_dict()

        comparison_rows.append(
            {
                "model": model_name,
                "sample_count": int(len(tmp)),
                "distinct_mechanisms": int(tmp["Behavioral Mechanism"].nunique()),
                "top_mechanism": top_mech,
                "top_mechanism_count": top_mech_count,
                "SD_count": int(variant_counts.get("SD", 0)),
                "SD_TYPO_count": int(variant_counts.get("SD_TYPO", 0)),
                "TYPO_count": int(variant_counts.get("TYPO", 0)),
                "SAFE_to_SAFE_count": int(transition_counts.get("SAFE_to_SAFE", 0)),
                "SAFE_to_UNSAFE_count": int(transition_counts.get("SAFE_to_UNSAFE", 0)),
                "UNSAFE_to_SAFE_count": int(transition_counts.get("UNSAFE_to_SAFE", 0)),
                "UNSAFE_to_UNSAFE_count": int(transition_counts.get("UNSAFE_to_UNSAFE", 0)),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("model")
    comparison_csv = OUTPUT_ROOT / "isap_all_models_comparison.csv"
    comparison_md = OUTPUT_ROOT / "isap_all_models_comparison.md"
    comparison_df.to_csv(comparison_csv, index=False)
    comparison_md.write_text(
        "# ISAP Cross-Model Comparison\n\n" + comparison_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved cross-model comparison -> {comparison_csv}")
    print(f"Saved cross-model comparison markdown -> {comparison_md}")


if __name__ == "__main__":
    main()