from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

MODEL_PATH = "liuhaotian/llava-v1.5-7b"
MODEL_BASE = None

BASE = Path("data")
QUESTIONS = BASE / "processed_questions" / "01-Illegal_Activitiy.json"
IMG_ROOT = Path("outputs/01-Illegal_Activitiy/masked")
OUT_DIR = Path("results_masked")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("SD", "Rephrased Question(SD)"),
    ("SD_TYPO", "Rephrased Question"),
    ("TYPO", "Rephrased Question"),
]


def run_variant(model, tokenizer, image_processor, variant_name, prompt_key, data, max_samples=None):
    image_folder = IMG_ROOT / variant_name
    out_file = OUT_DIR / f"01-Illegal_Activitiy_{variant_name}_answers.jsonl"

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
            images = process_images([image], image_processor, model.config).to(
                model.device, dtype=torch.float16
            )

            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
            conv.append_message(conv.roles[1], None)
            full_prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=256,
                    use_cache=True,
                )

            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            out.write(
                json.dumps(
                    {
                        "question_id": int(qid),
                        "image": f"{qid}.jpg",
                        "question": prompt,
                        "response": text,
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
    parser = argparse.ArgumentParser(description="Run LLaVA on masked Illegal Activity images.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for testing (e.g. 1 or 5).",
    )
    args = parser.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(MODEL_PATH)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH,
        MODEL_BASE,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device="cuda",
    )
    model = model.to("cuda")
    model.eval()

    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    for variant_name, prompt_key in VARIANTS:
        run_variant(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            variant_name=variant_name,
            prompt_key=prompt_key,
            data=data,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()