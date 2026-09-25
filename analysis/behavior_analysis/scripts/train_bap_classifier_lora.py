#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

TASK_SPECS = {
    "effect": {
        "label_candidates": ["behavioral_effect", "behavioral_effect_human", "effect"],
        "label_name": "Behavioral Effect",
        "valid_labels": [
            "Refusal Introduced",
            "Refusal Removed",
            "Assistance Increased",
            "Assistance Reduced",
            "Benign Reinterpretation",
            "No Meaningful Behavioral Change",
            "Partial Refusal Introduced",
            "Behavior Shift",
            "Mixed Behavioral Change",
        ],
        "output_prefix": "effect",
    },
    "mechanism": {
        "label_candidates": [
            "primary_driver",
            "primary_driver_human",
            "behavioral_mechanism",
            "behavioral_mechanism_human",
        ],
        "label_name": "Behavioral Mechanism",
        "valid_labels": [
            "Stable Refusal",
            "Stable Harmful Assistance",
            "Harmful Intent Ignored",
            "OCR Dependency",
            "Prompt Dependency",
            "Visual Grounding",
            "Image Sufficient",
            "Image Insufficient",
            "Benign Reinterpretation",
            "Operational Reinterpretation",
            "Mixed / Shared Influence",
            "Ambiguous Mechanism",
        ],
        "output_prefix": "mechanism",
    },
}


def find_label_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def build_context(row: pd.Series) -> str:
    def add(name: str, value: object) -> str:
        txt = "" if value is None else str(value).strip()
        return f"{name}: {txt}" if txt else ""

    parts = [
        add("Scenario", row.get("scenario")),
        add("Variant", row.get("variant")),
        add("Transition", row.get("transition")),
        add("Transition group", row.get("transition_group")),
        add("Original question", row.get("original_question")),
        add("Changed question", row.get("changed_question")),
        add("Key phrase", row.get("key_phrase")),
        add("Prompt", row.get("prompt")),
        add("Original response", row.get("original_response")),
        add("Masked response", row.get("masked_response")),
    ]
    return "\n".join([p for p in parts if p])


def prepare_dataframe(df: pd.DataFrame, label_col: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    out["context"] = out.apply(build_context, axis=1)
    out["text"] = out["context"]

    if label_col is not None and label_col in out.columns:
        out[label_col] = clean_series(out[label_col])
        out = out[out[label_col].notna()].copy()

    return out


def safe_split(df: pd.DataFrame, label_col: str, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[label_col] = clean_series(df[label_col])
    df = df[df[label_col].notna()].copy()

    counts = df[label_col].value_counts()
    rare_labels = counts[counts < 2].index.tolist()

    rare_df = df[df[label_col].isin(rare_labels)].copy()
    work_df = df[~df[label_col].isin(rare_labels)].copy()

    if len(work_df) == 0:
        raise ValueError("No rows left after removing rare labels.")

    stratify = None
    if work_df[label_col].nunique() > 1 and work_df[label_col].value_counts().min() >= 2:
        stratify = work_df[label_col]

    train_df, test_df = train_test_split(
        work_df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    if len(rare_df) > 0:
        train_df = pd.concat([train_df, rare_df], ignore_index=True)

    return train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True), test_df.reset_index(drop=True)


def build_tokenized_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    label_col: Optional[str] = None,
) -> Dataset:
    if label_col is not None and label_col in df.columns:
        ds_df = df[["text", label_col]].copy().rename(columns={label_col: "labels"})
    else:
        ds_df = df[["text"]].copy()

    ds = Dataset.from_pandas(ds_df, preserve_index=False)

    def tok(example):
        enc = tokenizer(example["text"], truncation=True, max_length=max_length)
        if "labels" in example:
            enc["labels"] = example["labels"]
        return enc

    ds = ds.map(tok, remove_columns=["text"])
    return ds


def load_model_and_tokenizer(base_model: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "single_label_classification"
    model.config.use_cache = False
    return model, tokenizer


def add_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float):
    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    return get_peft_model(model, cfg)


def _to_long_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.long()
    return torch.tensor(x, dtype=torch.long)


class SimpleCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features):
        input_ids = [_to_long_tensor(f["input_ids"]) for f in features]
        attention_mask = [_to_long_tensor(f["attention_mask"]) for f in features]

        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            ),
        }

        if "labels" in features[0]:
            batch["labels"] = torch.tensor(
                [int(f["labels"]) for f in features], dtype=torch.long
            )

        return batch


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(logits)
    return exps / exps.sum(axis=-1, keepdims=True)


@torch.no_grad()
def logits_from_texts(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int = 8,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    outs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        outs.append(out.logits.detach().float().cpu().numpy())

    return np.concatenate(outs, axis=0) if outs else np.zeros((0, model.config.num_labels))


def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Tuple[Dict[str, float], str, np.ndarray, List[str]]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
    }
    return metrics, report, cm, labels


def train_split(
    df: pd.DataFrame,
    task: str,
    base_model: str,
    output_dir: Path,
    test_size: float,
    seed: int,
    max_length: int,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    spec = TASK_SPECS[task]
    label_col = find_label_column(df, spec["label_candidates"])
    if label_col is None:
        raise ValueError(f"Could not find label column for task={task}. Tried: {spec['label_candidates']}")

    task_root = output_dir / task
    split_root = task_root / "split"
    split_root.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(df, label_col=label_col)
    train_df, test_df = safe_split(df, label_col, test_size=test_size, seed=seed)

    label_encoder = LabelEncoder()
    label_encoder.fit(df[label_col].astype(str).tolist())
    id2label = {i: lab for i, lab in enumerate(label_encoder.classes_)}
    label2id = {lab: i for i, lab in id2label.items()}

    model, tokenizer = load_model_and_tokenizer(base_model, num_labels=len(label_encoder.classes_))
    model = add_lora(model, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    model.config.id2label = id2label
    model.config.label2id = label2id

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["labels"] = train_df[label_col].map(label2id).astype(int)
    test_df["labels"] = test_df[label_col].map(label2id).astype(int)

    train_ds = build_tokenized_dataset(train_df, tokenizer, max_length, label_col="labels")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(split_root / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="epoch",
            report_to="none",
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            remove_unused_columns=False,
            save_total_limit=2,
        ),
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=SimpleCollator(tokenizer),
    )

    trainer.train()

    # Held-out evaluation, done manually to keep inference simple and memory-safe.
    test_logits = logits_from_texts(trainer.model, tokenizer, test_df["text"].tolist(), max_length=max_length)
    test_probs = softmax_np(test_logits)
    y_pred_ids = test_logits.argmax(axis=-1)
    y_pred = label_encoder.inverse_transform(y_pred_ids)
    y_true = test_df[label_col].astype(str).tolist()
    confs = test_probs.max(axis=-1)

    metrics, report, cm, labels = evaluate_predictions(y_true, y_pred)
    print("\nHeld-out classification metrics")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro precision: {metrics['macro_precision']:.3f}")
    print(f"Macro recall: {metrics['macro_recall']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}\n")
    print(report)

    heldout_out = test_df.copy()
    heldout_out[f"pred_{spec['output_prefix']}"] = y_pred
    heldout_out[f"pred_{spec['output_prefix']}_confidence"] = confs
    heldout_out.to_csv(split_root / "heldout_predictions.csv", index=False)

    trainer.model.save_pretrained(split_root / "adapter")
    tokenizer.save_pretrained(split_root / "adapter")

    save_json(
        split_root / "meta.json",
        {
            "task": task,
            "label_column": label_col,
            "labels": list(label_encoder.classes_),
            "base_model": base_model,
            "max_length": max_length,
        },
    )
    save_json(split_root / "metrics.json", metrics)
    save_text(split_root / "classification_report.txt", report)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(split_root / "confusion_matrix.csv", index=True)
    train_df.to_csv(split_root / "train_split.csv", index=False)
    test_df.to_csv(split_root / "test_split.csv", index=False)

    save_json(
        task_root / "run_config.json",
        {
            "mode": "split_train",
            "task": task,
            "base_model": base_model,
            "output_dir": str(output_dir),
            "test_size": test_size,
            "seed": seed,
            "max_length": max_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr": lr,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
    )

    print(f"\nSaved split model to: {split_root / 'adapter'}")


def train_full(
    df: pd.DataFrame,
    task: str,
    base_model: str,
    output_dir: Path,
    max_length: int,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    spec = TASK_SPECS[task]
    label_col = find_label_column(df, spec["label_candidates"])
    if label_col is None:
        raise ValueError(f"Could not find label column for task={task}. Tried: {spec['label_candidates']}")

    task_root = output_dir / task
    full_root = task_root / "full"
    full_root.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(df, label_col=label_col)

    label_encoder = LabelEncoder()
    label_encoder.fit(df[label_col].astype(str).tolist())
    id2label = {i: lab for i, lab in enumerate(label_encoder.classes_)}
    label2id = {lab: i for i, lab in id2label.items()}

    model, tokenizer = load_model_and_tokenizer(base_model, num_labels=len(label_encoder.classes_))
    model = add_lora(model, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    model.config.id2label = id2label
    model.config.label2id = label2id

    df = df.copy()
    df["labels"] = df[label_col].map(label2id).astype(int)
    full_ds = build_tokenized_dataset(df, tokenizer, max_length, label_col="labels")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(full_root / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="epoch",
            report_to="none",
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            remove_unused_columns=False,
            save_total_limit=2,
        ),
        train_dataset=full_ds,
        tokenizer=tokenizer,
        data_collator=SimpleCollator(tokenizer),
    )

    trainer.train()
    trainer.model.save_pretrained(full_root / "adapter")
    tokenizer.save_pretrained(full_root / "adapter")

    save_json(
        full_root / "meta.json",
        {
            "task": task,
            "label_column": label_col,
            "labels": list(label_encoder.classes_),
            "base_model": base_model,
            "max_length": max_length,
        },
    )
    df.to_csv(full_root / "full_train_data.csv", index=False)

    save_json(
        task_root / "run_config.json",
        {
            "mode": "full_train",
            "task": task,
            "base_model": base_model,
            "output_dir": str(output_dir),
            "max_length": max_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr": lr,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
    )

    print(f"Saved full model to: {full_root / 'adapter'}")


@torch.no_grad()
def predict(
    task: str,
    base_model: str,
    checkpoint_dir: Path,
    input_csv: Path,
    output_csv: Path,
    max_length: int,
):
    spec = TASK_SPECS[task]
    meta_path = checkpoint_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {checkpoint_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_names = meta["labels"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "adapter", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label_names),
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir / "adapter")
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    label_encoder = LabelEncoder()
    label_encoder.fit(label_names)

    df = pd.read_csv(input_csv, keep_default_na=False)
    df = prepare_dataframe(df, label_col=None)

    logits = logits_from_texts(model, tokenizer, df["text"].tolist(), max_length=max_length)
    probs = softmax_np(logits)
    pred_ids = logits.argmax(axis=-1)
    pred_labels = label_encoder.inverse_transform(pred_ids)
    pred_conf = probs.max(axis=-1)

    out = df.copy()
    out[f"pred_{spec['output_prefix']}"] = pred_labels
    out[f"pred_{spec['output_prefix']}_confidence"] = pred_conf
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Mistral LoRA classifier for BAP v2 labels.")
    parser.add_argument("--mode", choices=["split_train", "full_train", "predict"], required=True)
    parser.add_argument("--task", choices=["effect", "mechanism"], required=True)

    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)

    parser.add_argument("--base-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/behavior_analysis/runs/2026-07-15_mistral_cls_run"))

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=1024)

    parser.add_argument("--epochs", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode in {"split_train", "full_train"} and args.train_csv is None:
        raise ValueError("--train-csv is required for training modes")

    if args.mode == "predict":
        if args.input_csv is None:
            raise ValueError("--input-csv is required in predict mode")
        if args.output_csv is None:
            raise ValueError("--output-csv is required in predict mode")
        if args.checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required in predict mode")

    if args.mode == "split_train":
        df = pd.read_csv(args.train_csv, keep_default_na=False)
        train_split(
            df=df,
            task=args.task,
            base_model=args.base_model,
            output_dir=args.output_dir,
            test_size=args.test_size,
            seed=args.seed,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        return

    if args.mode == "full_train":
        df = pd.read_csv(args.train_csv, keep_default_na=False)
        train_full(
            df=df,
            task=args.task,
            base_model=args.base_model,
            output_dir=args.output_dir,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        return

    if args.mode == "predict":
        predict(
            task=args.task,
            base_model=args.base_model,
            checkpoint_dir=args.checkpoint_dir,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            max_length=args.max_length,
        )
        return


if __name__ == "__main__":
    main()