# Checkbox Detection — Writeup

## 1. Problem and constraints

Appraisal forms (URAR, Market Conditions, Manufactured Home) pack dozens of small square checkboxes next to table rules, typed text, and noise (scribbles, strike-throughs, watermarks, ✓ vs X). Only four sample pages are available, so a learned detector would need external or synthetic labels before it could beat a careful classical baseline.

## 2. Approach

Classical OpenCV pipeline in `app/detector.py`:

1. Grayscale + adaptive threshold (ink = white)
2. Morphological opening to subtract long horizontal/vertical rules
3. Connected components + size/aspect filters scaled to image width (`s = width / 1700`)
4. Square test on the un-cleaned binary: find four straight sides (rows/columns whose longest ink run spans ≥80% of the candidate), each covered end-to-end into the corners. Letters fail this — `o`, `e`, `a` have no straight sides, `M`/`N` lack horizontals, `D`/`O` have rounded corners. Running on the un-cleaned image means sides shared with table rules still count. The box is refined to those lines, which also trims X marks that overshoot the frame.
5. Surround check: the area just outside the box must be light, which rejects reverse-video text and bars (white-on-black / white-on-blue sidebars)
6. Hole proposals (second source): keep short H/V lines, dilate, and take empty interiors of checkbox size. Recovers boxes whose ink ring was eaten because a side is fused to a table rule. Kernels scale with page width; reverse-video holes are rejected by the surround check.
7. Fill check: ink density in a 25% inset interior → `is_checked`
8. IoU-based NMS across both proposal sources

An earlier version used a looser "ring" check (some ink along each bounding-box edge). Letters like `o`, `e`, `a`, `M` passed it, which produced hundreds of false positives per page. The straight-sides test removed them.

Tunables live in `app/config.py`. The FastAPI layer only decodes uploads and serializes results (or returns a debug overlay).

## 3. Alternatives considered

| Approach | Why not first |
|---|---|
| YOLO fine-tune | Needs labeled boxes; only 4 pages on hand |
| Hybrid CNN on crops | Good next step if fill threshold is brittle |
| VLM | Weak at tiny precise bboxes; slow, costly, nondeterministic |

## 4. Results

Labels under `eval/labels/` were **bootstrapped from detector output** and lightly reviewed, so `python -m eval.evaluate` is a **regression baseline**, not an independent accuracy claim. Re-label carefully before quoting absolute precision/recall.

Qualitative behavior on the four samples, checked by eye against full-page overlays:

| Sample | Boxes | Checked | Notes |
|---|---|---|---|
| `0.png` URAR | 118 | 37 | Ring + hole proposals; table-fused empties recovered; no letter FPs |
| `1.jpg` scribbled crop | 41 | 17 | ✓ mark classified checked. Still misses the faint/dotted Electricity box and the "No Zoning" box under the scribble |
| `2.png` Market Conditions | 48 | 12 | Boxes on light-blue shaded rows found; blue sidebar text rejected |
| `3.png` manufactured home | 79 | 27 | Hole pass recovered a couple of watermark/table-fused boxes |

Box sizes vary more across pages than page width predicts (≈57 px on `0`/`2`, ≈31 px on `3`, all 2550 px wide), so the size range is deliberately wide and the square test does the discriminating.

## 5. Tradeoffs

- **No training data vs accuracy** — classical CV ships immediately and is explainable; it will miss dotted/odd boxes and confuse heavy scribbles.
- **Relative size filters** — scale with page width; still sensitive to unusually large or tiny scans.
- **Fill threshold** — simple and fast; faint checks can fall below it; strike-throughs can push empty boxes over it.

## 6. Known limitations

- Skewed or low-resolution scans
- Boxes fused to table lines or each other
- Scribbles, strike-throughs, or watermarks crossing a box
- Non-square / dotted / rounded boxes
- Light marks below the fill threshold
- Thresholds tuned on four pages → likely overfit

## 7. Production next steps

Deskew; learned detector or CNN fill classifier; more labeled data; per-box confidence; request size limits (413); auth and rate limiting; structured logging and metrics; batching. See `TODO` comments in `app/main.py` and `app/detector.py`.
