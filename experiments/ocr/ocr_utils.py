from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@lru_cache(maxsize=4)
def get_reader(lang: tuple[str, ...] = ("en",), gpu: bool = False) -> easyocr.Reader:
    return easyocr.Reader(list(lang), gpu=gpu)


def detect_text(
    image_path: str | Path,
    lang: tuple[str, ...] = ("en",),
    gpu: bool = False,
    min_conf: float = 0.30,
) -> list[dict]:
    reader = get_reader(lang=lang, gpu=gpu)
    results = reader.readtext(str(image_path))

    detections: list[dict] = []
    for box, text, conf in results:
        conf = float(conf)
        if conf < min_conf:
            continue
        detections.append(
            {
                "box": [[float(x), float(y)] for x, y in box],
                "text": str(text),
                "conf": conf,
            }
        )
    return detections


def save_detections_json(
    detections: Sequence[dict],
    out_path: str | Path,
    image_path: str | Path | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "image_path": str(image_path) if image_path is not None else None,
        "num_detections": len(detections),
        "detections": list(detections),
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _to_int_points(box: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in box]


def draw_boxes(
    image_path: str | Path,
    detections: Sequence[dict],
    out_path: str | Path,
    outline: str = "red",
) -> Path:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for idx, det in enumerate(detections):
        pts = _to_int_points(det["box"])
        draw.polygon(pts, outline=outline, width=4)
        x0, y0 = pts[0]
        label = f'{idx}: {det["text"]} ({det["conf"]:.2f})'
        draw.text((x0, max(0, y0 - 18)), label, fill=outline)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def mask_detections(
    image_path: str | Path,
    detections: Sequence[dict],
    out_path: str | Path,
    indices: Optional[Iterable[int]] = None,
    mode: str = "average",
    blur_radius: int = 18,
) -> Path:
    """
    mode:
      - average: replace region with the region's average RGB
      - white:   replace region with white
      - blur:    blur only the selected regions
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    if indices is None:
        selected = set(range(len(detections)))
    else:
        selected = set(int(i) for i in indices)

    for idx, det in enumerate(detections):
        if idx not in selected:
            continue

        pts = _to_int_points(det["box"])
        region_mask = Image.new("L", img.size, 0)
        d = ImageDraw.Draw(region_mask)
        d.polygon(pts, fill=255)
        mask_np = np.array(region_mask) > 0

        if not mask_np.any():
            continue

        if mode == "blur":
            blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            img.paste(blurred, mask=region_mask)
        elif mode == "white":
            arr[mask_np] = np.array([255, 255, 255], dtype=np.uint8)
            img = Image.fromarray(arr)
        else:
            fill = arr[mask_np].mean(axis=0).astype(np.uint8)
            arr[mask_np] = fill
            img = Image.fromarray(arr)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path