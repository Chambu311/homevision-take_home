"""Precision / recall @ IoU 0.5 and fill accuracy on matched boxes.

Usage:
  python -m eval.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

from app.detector import decode_image, detect_checkboxes

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
LABELS = ROOT / "eval" / "labels"


def iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def match(
    preds: list[dict],
    gts: list[dict],
    iou_thresh: float = 0.5,
) -> tuple[int, int, int, int]:
    """Return tp, fp, fn, fill_correct among matched pairs."""
    used_gt: set[int] = set()
    tp = 0
    fill_correct = 0
    for pred in preds:
        best_i = -1
        best_iou = 0.0
        for i, gt in enumerate(gts):
            if i in used_gt:
                continue
            score = iou(pred["bbox"], gt["bbox"])
            if score > best_iou:
                best_iou = score
                best_i = i
        if best_i >= 0 and best_iou >= iou_thresh:
            used_gt.add(best_i)
            tp += 1
            if bool(pred["is_checked"]) == bool(gts[best_i]["is_checked"]):
                fill_correct += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn, fill_correct


def main() -> None:
    totals = {"tp": 0, "fp": 0, "fn": 0, "fill_correct": 0, "matched": 0}
    print(f"{'sample':<10} {'P':>6} {'R':>6} {'F1':>6} {'fill':>6} {'pred':>5} {'gt':>5}")
    for label_path in sorted(LABELS.glob("*.json")):
        stem = label_path.stem
        sample = next(SAMPLES.glob(f"{stem}.*"), None)
        if sample is None:
            print(f"{stem}: missing sample image")
            continue
        image = decode_image(sample.read_bytes())
        if image is None:
            print(f"{stem}: undecodable")
            continue
        preds = [
            {"bbox": list(d.bbox), "is_checked": d.is_checked}
            for d in detect_checkboxes(image)
        ]
        gts = json.loads(label_path.read_text()).get("boxes", [])
        tp, fp, fn, fill_correct = match(preds, gts)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        fill_acc = fill_correct / tp if tp else 0.0
        print(
            f"{stem:<10} {precision:6.3f} {recall:6.3f} {f1:6.3f} "
            f"{fill_acc:6.3f} {len(preds):5d} {len(gts):5d}"
        )
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["fill_correct"] += fill_correct
        totals["matched"] += tp

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    fill_acc = (
        totals["fill_correct"] / totals["matched"] if totals["matched"] else 0.0
    )
    print("-" * 52)
    print(
        f"{'ALL':<10} {precision:6.3f} {recall:6.3f} {f1:6.3f} "
        f"{fill_acc:6.3f}"
    )
    print(
        "Note: labels were bootstrapped from detector output and lightly reviewed; "
        "treat metrics as a regression baseline, not absolute truth."
    )


if __name__ == "__main__":
    main()
