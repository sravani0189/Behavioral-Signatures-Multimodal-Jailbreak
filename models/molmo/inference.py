from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

MODEL_ID = "allenai/Molmo-7B-D-0924"
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
        torch_dtype="auto",
        device_map="auto",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, processor


def run_variant(
    model,
    processor,
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

    generation_config = GenerationConfig(
        max_new_tokens=256,
        stop_strings="<|endoftext|>",
    )

    done = 0
    with out_file.open("w", encoding="utf-8") as out:
        for qid in sorted(data.keys(), key=lambda x: int(x)):
            item = data[qid]
            image_path = image_folder / f"{qid}.jpg"
            prompt = item[prompt_key]

            if not image_path.exists():
                print(f"[warn] Missing image: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")

            inputs = processor.process(
                images=[image],
                text=prompt,
            )

            # Move tensor inputs to the model device and add a batch dimension.
            inputs = {
                k: v.to(model.device).unsqueeze(0)
                for k, v in inputs.items()
                if torch.is_tensor(v)
            }

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    output = model.generate_from_batch(
                        inputs,
                        generation_config,
                        tokenizer=processor.tokenizer,
                    )

            generated_tokens = output[0, inputs["input_ids"].size(1):]
            response = processor.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            out.write(
                json.dumps(
                    {
                        "question_id": int(qid),
                        "image": f"{qid}.jpg",
                        "question": prompt,
                        "response": response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

            print(f"{variant_name}: {qid}")

            done += 1
            if max_samples is not None and done >= max_samples:
                print(f"[info] Reached max_samples={max_samples} for {variant_name}")
                break

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
    model, processor = load_model()

    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)

    for variant_name, prompt_key in VARIANTS:
        run_variant(
            model=model,
            processor=processor,
            image_root=image_root,
            out_dir=out_dir,
            variant_name=variant_name,
            prompt_key=prompt_key,
            data=data,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()