"""Tunable parameters for the checkbox detection pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    """Relative sizes are scaled by s = image_width / reference_width."""

    reference_width: float = 1700.0

    # Geometry filter (pixels at reference width, then × s).
    # Sample boxes measure ~21-38 px at reference width, so the range is wide.
    box_size_min: float = 14.0
    box_size_max: float = 48.0
    aspect_min: float = 0.7
    aspect_max: float = 1.4

    # Adaptive threshold
    adaptive_block_size: int = 31  # must be odd
    adaptive_c: float = 15.0

    # Long line removal kernel length (px at reference width, then × s)
    line_kernel_length: float = 40.0

    # Hole proposals: short H/V kernel (px at reference width, then × s).
    # Finds empty square interiors even when a side is fused to a table rule.
    hole_kernel_length: float = 15.0

    # Square test: each side must be a straight line spanning the box.
    # A row/column counts as a side if its longest ink run is >= this
    # fraction of the candidate's width/height.
    side_run_ratio: float = 0.8
    # Fraction of the side (between corners) that must be inked.
    side_coverage_min: float = 0.9
    # The first and last corner_size fraction of every side must be fully
    # inked; rejects rounded glyphs like D, O, 0.
    corner_size: float = 0.1
    # Median grayscale just outside the box; rejects reverse-video text and bars.
    surround_min_gray: float = 140.0

    # Interior fill: inset fraction of the box; density above threshold => checked
    fill_inset: float = 0.25
    fill_threshold: float = 0.09

    # NMS
    nms_iou_threshold: float = 0.35


DEFAULT_CONFIG = DetectorConfig()
