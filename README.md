cat > README.md <<'EOF'
# Beyond Attack Success Rate: Behavioral Signatures for Evaluating Multimodal Jailbreak Robustness

This repository contains code and analysis artifacts for our work on
**Behavioral Signatures**, a behavior-centric framework for evaluating
multimodal jailbreak robustness beyond aggregate Attack Success Rate (ASR).

The framework analyzes paired responses to original multimodal jailbreak inputs
and OCR-masked counterfactual inputs using the **Behavior Analysis Protocol
(BAP)**. Each response pair is characterized along three dimensions:

- **Behavioral Effect** — how the observable response behavior changes.
- **Behavioral Attribution** — the response pattern most consistent with the
  observed change.
- **Safety Transition** — whether the response changes between SAFE and UNSAFE.

These sample-level annotations are aggregated into model-level Behavioral
Signatures for cross-model comparison.

## Overview

The evaluation uses the **Illegal Activity** subset of
[MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench), including the
SD, SD_TYPO, and TYPO variants.

For each benchmark instance, OCR-accessible text is detected and masked to
construct a counterfactual input. Responses to the original and masked inputs
are then compared using BAP.

The current study evaluates ten multimodal language models:

- GPT-5.6 Sol
- GPT-5.6 Terra
- InternVL2.5
- Llama-3.2 Vision
- MiniCPM-V 2.6
- Molmo-7B
- Ovis2.5-9B
- Phi-4 Multimodal
- Pixtral-12B
- Qwen2.5-VL

The final behavioral analysis contains **2,877 structured response-pair
annotations**.

## Repository Structure

```text
.
├── analysis/
│   ├── behavior_analysis/
│   │   ├── auto_annotations/
│   │   ├── data/
│   │   ├── outputs/
│   │   └── scripts/
│   ├── cross_transition/
│   └── ...
├── experiments/
│   ├── ocr/
│   └── ocr_masking_transition/
├── models/
│   ├── gpt56_sol/
│   ├── gpt-5.6-terra/
│   ├── internvl25/
│   ├── llama32_vision/
│   ├── minicpmv26/
│   ├── molmo/
│   ├── ovis25_9b/
│   ├── phi4mm/
│   ├── pixtral12b/
│   └── qwen25_vl/
├── scripts/
├── evaluation.py
├── evaluation_human_guideline.py
├── run_illegal_activity_all.py
└── run_illegal_activity_masked.py
