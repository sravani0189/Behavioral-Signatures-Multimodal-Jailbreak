from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_utils import detect_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect OCR text in an image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--min-conf", type=float, default=0.30, help="Minimum confidence")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    detections = detect_text(
        args.image,
        gpu=args.gpu,
        min_conf=args.min_conf,
    )

    print(f"Detected regions: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"[{i}] TEXT: {det['text']}")
        print(f"    CONF: {det['conf']:.4f}")
        print(f"    BOX : {det['box']}")
        print("-" * 50)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")
        print(f"Saved JSON to: {out_path}")


if __name__ == "__main__":
    main()