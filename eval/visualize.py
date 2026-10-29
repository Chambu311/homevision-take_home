"""Draw predictions (and optional ground truth) onto sample images.

Usage:
  python -m eval.visualize
  python -m eval.visualize --sample 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from app.detector import Detection, annotate_detections, decode_image, detect_checkboxes

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
LABELS = ROOT / "eval" / "labels"
OUT = ROOT / "eval" / "out"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize checkbox detections")
    parser.add_argument("--sample", help="Sample stem (e.g. 0). Default: all.")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(SAMPLES.glob("*"))
    if args.sample is not None:
        paths = [p for p in paths if p.stem == args.sample]
        if not paths:
            raise SystemExit(f"No sample matching stem={args.sample}")

    for path in paths:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        image = decode_image(path.read_bytes())
        if image is None:
            print(f"skip undecodable {path.name}")
            continue

        detections = detect_checkboxes(image)
        pred_img = annotate_detections(image, detections)
        pred_path = OUT / f"{path.stem}_pred.jpg"
        cv2.imwrite(str(pred_path), pred_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print(f"{path.name}: {len(detections)} boxes -> {pred_path}")

        label_path = LABELS / f"{path.stem}.json"
        if label_path.exists():
            data = json.loads(label_path.read_text())
            gt = [
                Detection(
                    bbox=tuple(b["bbox"]),  # type: ignore[arg-type]
                    is_checked=bool(b["is_checked"]),
                )
                for b in data.get("boxes", [])
            ]
            # Blue = ground-truth outline for comparison
            gt_img = image.copy()
            for det in gt:
                x1, y1, x2, y2 = det.bbox
                color = (255, 128, 0) if det.is_checked else (255, 200, 0)
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 2)
            # Overlay predictions on top
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                color = (0, 0, 255) if det.is_checked else (0, 200, 0)
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 1)
            cmp_path = OUT / f"{path.stem}_compare.jpg"
            cv2.imwrite(str(cmp_path), gt_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            print(f"  compare -> {cmp_path}")


if __name__ == "__main__":
    main()
