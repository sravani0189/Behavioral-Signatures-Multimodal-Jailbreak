from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

MODEL_ID = "OpenGVLab/InternVL2_5-8B"
SCENARIO = "01-Illegal_Activitiy"

VARIANTS = [
    ("SD", "Rephrased Question(SD)"),
    ("SD_TYPO", "Rephrased Question"),
    ("TYPO", "Rephrased Question"),
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Compatibility patch for newer Transformers behavior
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = property(
        lambda self: getattr(self, "_tied_weights_keys", {})
    )


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_model():
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype="auto",
    ).cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
    )
    return model, tokenizer


def run_variant(
    model,
    tokenizer,
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

    generation_config = dict(
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
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

            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            question = f"<image>\n{prompt}"

            with torch.inference_mode():
                response = model.chat(tokenizer, pixel_values, question, generation_config)

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
    model, tokenizer = load_model()

    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)

    for variant_name, prompt_key in VARIANTS:
        run_variant(
            model=model,
            tokenizer=tokenizer,
            image_root=image_root,
            out_dir=out_dir,
            variant_name=variant_name,
            prompt_key=prompt_key,
            data=data,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()