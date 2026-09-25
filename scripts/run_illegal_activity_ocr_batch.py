from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# allow importing from experiments/ocr
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments" / "ocr"))

from ocr_utils import detect_text, draw_boxes, mask_detections, save_detections_json

SCENARIO = "01-Illegal_Activitiy"
VARIANTS = ["SD", "SD_TYPO", "TYPO"]


def detect_with_fallback(image_path: Path, device: str, min_conf: float):
    """
    device:
      - gpu: force GPU, fail if unavailable
      - cpu: force CPU
      - auto: try GPU first, then fall back to CPU
    """
    if device == "cpu":
        return detect_text(image_path, gpu=False, min_conf=min_conf), "cpu"

    if device == "gpu":
        return detect_text(image_path, gpu=True, min_conf=min_conf), "gpu"

    # auto
    try:
        detections = detect_text(image_path, gpu=True, min_conf=min_conf)
        return detections, "gpu"
    except Exception as e:
        print(f"[warn] GPU OCR failed for {image_path.name}: {e}")
        print("[warn] Falling back to CPU OCR...")
        detections = detect_text(image_path, gpu=False, min_conf=min_conf)
        return detections, "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="OCR device choice",
    )
    parser.add_argument("--min-conf", type=float, default=0.30)
    args = parser.parse_args()

    qa_file = Path(f"questions_with_answers/{SCENARIO}.json")
    image_root = Path(f"data/imgs/MM-SafetyBench(imgs)/{SCENARIO}")
    output_root = Path(f"outputs/{SCENARIO}")

    json_root = output_root / "ocr_json"
    vis_root = output_root / "annotated"
    mask_root = output_root / "masked"

    json_root.mkdir(parents=True, exist_ok=True)
    vis_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    data = json.loads(qa_file.read_text(encoding="utf-8"))

    rows = []

    for qid, item in data.items():
        for variant in VARIANTS:
            image_path = image_root / variant / f"{qid}.jpg"

            if not image_path.exists():
                print(f"Missing image: {image_path}")
                continue

            try:
                detections, used_device = detect_with_fallback(
                    image_path=image_path,
                    device=args.device,
                    min_conf=args.min_conf,
                )
            except Exception as e:
                print(f"[error] OCR failed completely for {image_path}: {e}")
                continue

            json_path = json_root / variant / f"{qid}.json"
            vis_path = vis_root / variant / f"{qid}.jpg"
            mask_path = mask_root / variant / f"{qid}.jpg"

            save_detections_json(detections, json_path, image_path)
            draw_boxes(image_path, detections, vis_path)
            mask_detections(image_path, detections, mask_path, mode="average")

            rows.append(
                {
                    "qid": qid,
                    "variant": variant,
                    "device": used_device,
                    "image_path": str(image_path),
                    "ocr_count": len(detections),
                    "ocr_json": str(json_path),
                    "annotated": str(vis_path),
                    "masked": str(mask_path),
                }
            )

            print(f"{variant}/{qid} -> OCR regions: {len(detections)} ({used_device})")

    csv_path = output_root / "manifest.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "qid",
                "variant",
                "device",
                "image_path",
                "ocr_count",
                "ocr_json",
                "annotated",
                "masked",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("DONE")
    print("Manifest:", csv_path)
    print("Total samples:", len(rows))


if __name__ == "__main__":
    main()