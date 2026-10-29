"""Unit tests for the OpenCV checkbox detector on a synthetic image."""

from __future__ import annotations

import cv2
import numpy as np

from app.detector import detect_checkboxes


def _synthetic_form() -> np.ndarray:
    """White page (~reference width) with two empty boxes and two X-marked boxes."""
    # Width near config.reference_width so size filters stay in range
    img = np.full((900, 1700, 3), 255, dtype=np.uint8)
    boxes = [
        (200, 200, 24, False),   # empty
        (280, 200, 24, True),    # checked with X
        (200, 280, 24, True),    # checked with X
        (280, 280, 24, False),   # empty
    ]
    for x, y, size, checked in boxes:
        cv2.rectangle(img, (x, y), (x + size, y + size), (0, 0, 0), 2)
        if checked:
            cv2.line(img, (x + 4, y + 4), (x + size - 4, y + size - 4), (0, 0, 0), 2)
            cv2.line(img, (x + size - 4, y + 4), (x + 4, y + size - 4), (0, 0, 0), 2)
    return img


def test_detects_synthetic_checkboxes() -> None:
    image = _synthetic_form()
    detections = detect_checkboxes(image)
    assert len(detections) >= 2

    checked = [d for d in detections if d.is_checked]
    unchecked = [d for d in detections if not d.is_checked]
    assert len(checked) >= 1
    assert len(unchecked) >= 1

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        assert x2 > x1 and y2 > y1


def test_empty_image_returns_no_boxes() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert detect_checkboxes(image) == []


def test_detects_box_fused_to_table_rule() -> None:
    """A checkbox whose bottom side is a long table rule should still be found."""
    img = np.full((900, 1700, 3), 255, dtype=np.uint8)
    cv2.line(img, (80, 400), (1620, 400), (0, 0, 0), 2)
    x, y, size = 300, 376, 24  # bottom edge sits on the table rule
    cv2.rectangle(img, (x, y), (x + size, y + size), (0, 0, 0), 2)

    detections = detect_checkboxes(img)
    matched = [
        d
        for d in detections
        if abs(d.bbox[0] - x) < 8 and abs(d.bbox[1] - y) < 8
    ]
    assert matched, f"expected fused box near {(x, y)}, got {detections}"
