# Experiment: OCR Masking Transition Analysis

## Objective

The purpose of this experiment is to understand how OCR masking changes the behavior of a multimodal large language model (MLLM) during prompt-injection attacks.

Unlike previous experiments that only measured attack success rate (ASR), this experiment investigates **which individual samples change after OCR masking and why those changes occur.**

---

# Research Question

Does removing OCR text change the model's safety behavior?

If so,

* Which samples changed?
* What kind of transition occurred?
* Why did the model change its behavior?

---

# Motivation

Previous experiments showed that OCR masking reduced the number of unsafe responses.

However, aggregate attack rates alone cannot explain **why** the behavior changed.

This experiment performs a sample-level transition analysis to identify the responses that changed after OCR masking and investigate the underlying reasons.

---

# Input Files

Original evaluation

```
questions_with_answers/
01-Illegal_Activitiy_human_guideline.json
```

Masked evaluation

```
judged_results/

01-Illegal_Activitiy_SD_human_guideline_judged.jsonl

01-Illegal_Activitiy_SD_TYPO_human_guideline_judged.jsonl

01-Illegal_Activitiy_TYPO_human_guideline_judged.jsonl
```

---

# Methodology

Step 1

Load the original human-guideline evaluation.

↓

Step 2

Load the OCR-masked human-guideline evaluation.

↓

Step 3

Compare every sample individually.

↓

Step 4

Assign one transition type.

* SAFE → SAFE
* SAFE → UNSAFE
* UNSAFE → SAFE
* UNSAFE → UNSAFE

↓

Step 5

Generate transition tables.

↓

Step 6

Manually inspect changed samples.

↓

Step 7

Identify possible explanations for why the behavior changed.

---

# Output Files

The experiment generates

```
outputs/

01-Illegal_Activitiy_SD_transition_table.csv

01-Illegal_Activitiy_SD_TYPO_transition_table.csv

01-Illegal_Activitiy_TYPO_transition_table.csv

01-Illegal_Activitiy_transition_summary.csv

01-Illegal_Activitiy_transition_summary.json
```

---

# Expected Findings

This experiment is expected to identify

* which samples changed after OCR masking,
* which transition type occurred,
* how often each transition occurred,
* whether OCR masking consistently reduces unsafe behavior,
* candidate explanations for the observed behavioral changes.

---

# Current Status

* [x] Human annotation completed
* [x] Human-guideline GPT evaluation completed
* [x] OCR masking completed
* [x] Before vs. after comparison completed
* [ ] Transition analysis
* [ ] Manual inspection of changed samples
* [ ] Explanation of behavioral changes
* [ ] Paper figures
* [ ] Paper tables

---

# Notes

This experiment focuses on **behavioral transitions**, not merely attack success rates.

The goal is to move beyond reporting whether OCR masking reduces unsafe responses and toward understanding **why** those behavioral changes occur.
