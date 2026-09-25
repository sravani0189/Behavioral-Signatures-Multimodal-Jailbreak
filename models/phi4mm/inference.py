from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, GenerationConfig

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
SCENARIO = "01-Illegal_Activitiy"

VARIANTS = [
    ("SD", "Rephrased Question(SD)"),
    ("SD_TYPO", "Rephrased Question"),
    ("TYPO", "Rephrased Question"),
]


def load_model():
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
    )

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation={"": "eager"},
    ).to("cuda")

    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")

    generation_config = GenerationConfig.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor, generation_config


def build_prompt(prompt: str) -> str:
    return f"<|user|><|image_1|>{prompt}<|end|><|assistant|>"


def run_variant(
    model,
    processor,
    generation_config,
    image_root: Path,
    out_dir: Path,
    variant_name: str,
    prompt_key: str,
    data: dict,
    max_samples=None,
):
    image_folder = image_root / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{SCENARIO}_{variant_name}_answers.jsonl"

    completed = set()
    if out_file.exists():
        with out_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        completed.add(int(row["question_id"]))
                    except Exception:
                        pass

    done = 0
    out = out_file.open("a", encoding="utf-8")
    try:
        for qid in sorted(data.keys(), key=lambda x: int(x)):
            qid_int = int(qid)
            if qid_int in completed:
                print(f"[skip] {variant_name}: {qid}")
                continue

            item = data[qid]
            image_path = image_folder / f"{qid}.jpg"
            prompt = item[prompt_key]

            if not image_path.exists():
                print(f"[warn] Missing image: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            full_prompt = build_prompt(prompt)

            inputs = processor(
                text=full_prompt,
                images=image,
                return_tensors="pt",
            ).to("cuda")

            with torch.inference_mode():
                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    generation_config=generation_config,
                )

            generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
            response = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            row = {
                "question_id": qid_int,
                "image": f"{qid}.jpg",
                "question": prompt,
                "response": response,
            }

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            print(f"{variant_name}: {qid}")
            done += 1
            if max_samples is not None and done >= max_samples:
                print(f"[info] Reached max_samples={max_samples} for {variant_name}")
                break
    finally:
        out.close()

    print(f"Saved {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Root folder containing SD / SD_TYPO / TYPO subfolders.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Where to write JSONL outputs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    questions_file = root / "data" / "processed_questions" / f"{SCENARIO}.json"
    data = json.loads(questions_file.read_text(encoding="utf-8"))

    model, processor, generation_config = load_model()

    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)

    for variant_name, prompt_key in VARIANTS:
        run_variant(
            model=model,
            processor=processor,
            generation_config=generation_config,
            image_root=image_root,
            out_dir=out_dir,
            variant_name=variant_name,
            prompt_key=prompt_key,
            data=data,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()