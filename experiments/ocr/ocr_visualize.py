from __future__ import annotations

import argparse
from pathlib import Path

from ocr_utils import detect_text, draw_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw OCR boxes on an image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save annotated image")
    parser.add_argument("--min-conf", type=float, default=0.30, help="Minimum confidence")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    detections = detect_text(
        args.image,
        gpu=args.gpu,
        min_conf=args.min_conf,
    )
    out = draw_boxes(args.image, detections, args.output)
    print(f"Saved annotated image to: {out}")


if __name__ == "__main__":
    main()