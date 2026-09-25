from __future__ import annotations

import argparse
from pathlib import Path

from ocr_utils import detect_text, mask_detections


def parse_indices(value: str | None):
    if value is None or value.strip().lower() == "all":
        return None
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mask OCR text regions in an image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save masked image")
    parser.add_argument("--indices", default="all", help="Comma-separated indices to mask, or 'all'")
    parser.add_argument(
        "--mode",
        choices=["average", "white", "blur"],
        default="average",
        help="Masking mode",
    )
    parser.add_argument("--min-conf", type=float, default=0.30, help="Minimum confidence")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    detections = detect_text(
        args.image,
        gpu=args.gpu,
        min_conf=args.min_conf,
    )

    selected = parse_indices(args.indices)
    out = mask_detections(
        args.image,
        detections,
        args.output,
        indices=selected,
        mode=args.mode,
    )
    print(f"Saved masked image to: {out}")


if __name__ == "__main__":
    main()