"""Classical OpenCV checkbox detector.

Pure functions: image in, boxes out. No FastAPI imports.

TODO(production):
- Deskew via Hough lines / minAreaRect before detection.
- Replace brittle fill threshold with a small CNN on proposed crops.
- Emit a confidence score per box for downstream filtering / VLM fallback.
- Support batch detection and structured logging of intermediate stats.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import DEFAULT_CONFIG, DetectorConfig


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    is_checked: bool


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode image bytes to BGR ndarray, or None if undecodable."""
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


def detect_checkboxes(
    image: np.ndarray,
    config: DetectorConfig = DEFAULT_CONFIG,
) -> list[Detection]:
    """Detect checkboxes and classify each as checked / unchecked."""
    if image is None or image.size == 0:
        return []

    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    height, width = gray.shape[:2]
    scale = width / config.reference_width

    binary = _adaptive_binary(gray, config)
    cleaned = _remove_long_lines(binary, scale, config)
    ring_candidates = _geometry_candidates(cleaned, scale, config)
    hole_candidates = _hole_candidates(binary, scale, config)

    detections: list[Detection] = []
    for x, y, w, h in ring_candidates:
        det = _to_detection(binary, gray, x, y, w, h, scale, config, require_square=True)
        if det is not None:
            detections.append(det)
    for x, y, w, h in hole_candidates:
        det = _to_detection(binary, gray, x, y, w, h, scale, config, require_square=False)
        if det is not None:
            detections.append(det)

    return _nms(detections, config.nms_iou_threshold)


def annotate_detections(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Draw boxes: green = unchecked, red = checked."""
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = (0, 0, 255) if det.is_checked else (0, 200, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    return canvas


def _adaptive_binary(gray: np.ndarray, config: DetectorConfig) -> np.ndarray:
    block = config.adaptive_block_size
    if block % 2 == 0:
        block += 1
    # Inverted so ink is white (255)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        config.adaptive_c,
    )


def _remove_long_lines(
    binary: np.ndarray,
    scale: float,
    config: DetectorConfig,
) -> np.ndarray:
    """Subtract long horizontal/vertical rules; short checkbox edges survive."""
    length = max(15, int(round(config.line_kernel_length * scale)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    lines = cv2.bitwise_or(horizontal, vertical)
    cleaned = cv2.subtract(binary, lines)

    # Light close to reconnect box corners broken by line subtraction
    reconnect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, reconnect)


def _geometry_candidates(
    cleaned: np.ndarray,
    scale: float,
    config: DetectorConfig,
) -> list[tuple[int, int, int, int]]:
    # Loose pre-filter: the component may be a mark overshooting its box or a
    # box missing a side that was removed with a table rule.
    min_size = config.box_size_min * scale * 0.6
    max_size = config.box_size_max * scale * 1.4

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    candidates: list[tuple[int, int, int, int]] = []
    for i in range(1, num_labels):  # skip background
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w < min_size or h < min_size or w > max_size or h > max_size:
            continue
        aspect = w / h if h else 0.0
        if aspect < config.aspect_min * 0.8 or aspect > config.aspect_max * 1.25:
            continue
        candidates.append((x, y, w, h))
    return candidates


def _hole_candidates(
    binary: np.ndarray,
    scale: float,
    config: DetectorConfig,
) -> list[tuple[int, int, int, int]]:
    """Empty square interiors enclosed by H/V ink, including table-fused boxes.

    Adapted from the hole/line approach in checkbox_detection_opencv, but
    kernels and size filters scale with image width, and the page exterior is
    skipped. Uses the adaptive binary so JPEG/shaded pages stay consistent.
    """
    k = max(8, int(round(config.hole_kernel_length * scale)))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    h_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_h)
    v_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_v)
    h_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_h)
    v_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_v)
    lines = cv2.bitwise_or(
        cv2.bitwise_and(h_open, h_close),
        cv2.bitwise_and(v_open, v_close),
    )
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)

    holes = cv2.bitwise_not(lines)
    _num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        holes, connectivity=8
    )

    min_size = config.box_size_min * scale
    max_size = config.box_size_max * scale
    img_h, img_w = binary.shape[:2]
    candidates: list[tuple[int, int, int, int]] = []
    # Label 0 is the line background; skip it. Size filter drops the page hole.
    for i in range(1, stats.shape[0]):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w < min_size or h < min_size or w > max_size or h > max_size:
            continue
        aspect = w / h if h else 0.0
        if aspect < config.aspect_min or aspect > config.aspect_max:
            continue
        if x <= 1 or y <= 1 or x + w >= img_w - 1 or y + h >= img_h - 1:
            continue
        candidates.append((x, y, w, h))
    return candidates


def _to_detection(
    binary: np.ndarray,
    gray: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    scale: float,
    config: DetectorConfig,
    *,
    require_square: bool,
) -> Detection | None:
    box = _square_box(binary, x, y, w, h, config)
    if box is None:
        if require_square:
            return None
        box = (x, y, x + w, y + h)
        # Hole interiors without a full square ring still need 3 inked walls
        # so letter counters (e.g. "Att.") don't sneak through.
        if not _has_enough_walls(binary, box, min_sides=3):
            return None
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    min_size = config.box_size_min * scale
    max_size = config.box_size_max * scale
    if not (min_size <= bw <= max_size and min_size <= bh <= max_size):
        return None
    if bh == 0 or not (config.aspect_min <= bw / bh <= config.aspect_max):
        return None
    if not _surroundings_are_light(gray, box, config):
        return None
    is_checked = _fill_check(binary, bx1, by1, bw, bh, config)
    return Detection(bbox=box, is_checked=is_checked)


def _longest_runs(mask: np.ndarray) -> np.ndarray:
    """Length of the longest run of True values in each row."""
    run = np.zeros(mask.shape[0], dtype=np.int32)
    best = np.zeros(mask.shape[0], dtype=np.int32)
    for col in mask.T:
        run = np.where(col, run + 1, 0)
        np.maximum(best, run, out=best)
    return best


def _square_box(
    binary: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    config: DetectorConfig,
) -> tuple[int, int, int, int] | None:
    """Find four straight sides around a candidate and return the refined box.

    Runs on the un-cleaned binary so sides shared with table rules still count.
    Letters fail because they lack straight lines spanning the full width and
    height on all four sides.
    """
    img_h, img_w = binary.shape[:2]
    pad = max(2, int(round(0.2 * max(w, h))))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
    roi = binary[y1:y2, x1:x2] > 0

    long_rows = np.flatnonzero(_longest_runs(roi) >= config.side_run_ratio * w)
    long_cols = np.flatnonzero(_longest_runs(roi.T) >= config.side_run_ratio * h)
    if len(long_rows) < 2 or len(long_cols) < 2:
        return None

    # Pick the lines nearest the component's own edges, not the outermost ones,
    # so a neighbouring table rule inside the padding is ignored.
    top = int(long_rows[np.argmin(np.abs(long_rows - (y - y1)))])
    bottom = int(long_rows[np.argmin(np.abs(long_rows - (y - y1 + h - 1)))])
    left = int(long_cols[np.argmin(np.abs(long_cols - (x - x1)))])
    right = int(long_cols[np.argmin(np.abs(long_cols - (x - x1 + w - 1)))])
    if bottom - top < 4 or right - left < 4:
        return None

    side_len = min(bottom - top, right - left) + 1
    t = max(1, side_len // 12)
    sides = (
        roi[top : top + t + 1, left : right + 1].any(axis=0),
        roi[bottom - t : bottom + 1, left : right + 1].any(axis=0),
        roi[top : bottom + 1, left : left + t + 1].any(axis=1),
        roi[top : bottom + 1, right - t : right + 1].any(axis=1),
    )
    # Every side must reach both corners; rounded glyphs (D, O, 0) don't.
    end = max(1, int(round(config.corner_size * side_len)))
    for side in sides:
        if float(side.mean()) < config.side_coverage_min:
            return None
        if not (side[:end].all() and side[-end:].all()):
            return None

    return (x1 + left, y1 + top, x1 + right + 1, y1 + bottom + 1)


def _has_enough_walls(
    binary: np.ndarray,
    box: tuple[int, int, int, int],
    min_sides: int = 3,
    coverage: float = 0.65,
) -> bool:
    x1, y1, x2, y2 = box
    roi = binary[y1:y2, x1:x2] > 0
    if min(roi.shape) < 4:
        return False
    t = max(1, min(roi.shape) // 12)
    scores = (
        float(roi[:t, :].mean()),
        float(roi[-t:, :].mean()),
        float(roi[:, :t].mean()),
        float(roi[:, -t:].mean()),
    )
    return sum(s >= coverage for s in scores) >= min_sides


def _surroundings_are_light(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    config: DetectorConfig,
) -> bool:
    """Median of a ring just outside the box; rejects reverse-video text and bars."""
    x1, y1, x2, y2 = box
    img_h, img_w = gray.shape[:2]
    m = max(2, (x2 - x1) // 5)
    ox1, oy1 = max(0, x1 - m), max(0, y1 - m)
    ox2, oy2 = min(img_w, x2 + m), min(img_h, y2 + m)
    outer = gray[oy1:oy2, ox1:ox2]
    mask = np.ones(outer.shape, dtype=bool)
    mask[y1 - oy1 : y2 - oy1, x1 - ox1 : x2 - ox1] = False
    ring = outer[mask]
    return ring.size > 0 and float(np.median(ring)) >= config.surround_min_gray


def _fill_check(
    binary: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    config: DetectorConfig,
) -> bool:
    """Ink density in the inset interior; above threshold => checked."""
    inset_x = max(1, int(round(w * config.fill_inset)))
    inset_y = max(1, int(round(h * config.fill_inset)))
    img_h, img_w = binary.shape[:2]

    ix1 = min(x + inset_x, img_w - 1)
    iy1 = min(y + inset_y, img_h - 1)
    ix2 = max(x + w - inset_x, ix1 + 1)
    iy2 = max(y + h - inset_y, iy1 + 1)
    ix2 = min(ix2, img_w)
    iy2 = min(iy2, img_h)

    interior = binary[iy1:iy2, ix1:ix2]
    if interior.size == 0:
        return False
    density = float(np.count_nonzero(interior)) / float(interior.size)
    return density > config.fill_threshold


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    if not detections:
        return []
    # Prefer larger boxes when overlapping (more complete ring)
    ordered = sorted(
        detections,
        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
        reverse=True,
    )
    kept: list[Detection] = []
    for det in ordered:
        if any(_iou(det.bbox, k.bbox) >= iou_threshold for k in kept):
            continue
        kept.append(det)
    # Stable reading order: top-to-bottom, left-to-right
    kept.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
    return kept
