from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "mistral-community/pixtral-12b"
SCENARIO = "01-Illegal_Activitiy"

VARIANTS = [
    ("SD", "Rephrased Question(SD)"),
    ("SD_TYPO", "Rephrased Question"),
    ("TYPO", "Rephrased Question"),
]


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    return model, processor


def build_chat(prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }
    ]


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

    completed = set()
    if out_file.exists():
        with out_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    completed.add(int(row["question_id"]))
                except Exception:
                    pass

    done = 0
    with out_file.open("a", encoding="utf-8") as out:
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
            chat = build_chat(prompt)

            prompt_text = processor.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=prompt_text,
                images=[image],
                return_tensors="pt",
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )

            response = processor.batch_decode(
                output_ids,
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

    print(f"Saved {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
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