from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

MODEL_PATH = "ATH-MaaS/Ovis2.5-9B"
SCENARIO = "01-Illegal_Activitiy"

VARIANTS = [
    ("SD", "Rephrased Question(SD)"),
    ("SD_TYPO", "Rephrased Question"),
    ("TYPO", "Rephrased Question"),
]


def load_model():
    # Official Ovis2.5 quick-inference path uses AutoModelForCausalLM + trust_remote_code.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda().eval()
    return model


def build_messages(prompt: str, image: Image.Image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run_variant(
    model,
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
            messages = build_messages(prompt, image)

            # Ovis2.5 official flow
            input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            input_ids = input_ids.cuda()
            pixel_values = pixel_values.cuda().to(model.dtype) if pixel_values is not None else None
            grid_thws = grid_thws.cuda() if grid_thws is not None else None

            with torch.inference_mode():
                outputs = model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    enable_thinking=False,
                    max_new_tokens=256,
                    do_sample=False,
                    eos_token_id=model.text_tokenizer.eos_token_id,
                    pad_token_id=model.text_tokenizer.pad_token_id,
                )

            response = model.text_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            ).strip()

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

    model = load_model()
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)

    for variant_name, prompt_key in VARIANTS:
        run_variant(
            model=model,
            image_root=image_root,
            out_dir=out_dir,
            variant_name=variant_name,
            prompt_key=prompt_key,
            data=data,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()