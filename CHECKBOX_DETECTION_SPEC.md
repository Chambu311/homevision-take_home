# Checkbox Detection API: Project Spec

Backend take-home for HomeVision. Build an HTTP service that finds checkboxes in appraisal document images and classifies each as checked or unchecked.

---

## 1. Goal

Design and implement a system that:

1. **Detects the location** of every checkbox (filled and unfilled) in a document image.
2. **Classifies each checkbox** as filled (`is_checked: true`) or unfilled (`is_checked: false`).

Perfect accuracy is not expected. Reasoning, tradeoffs and clear communication matter as much as the result.

## 2. Deliverables

- [ ] Working HTTP API with `POST /detect`
- [ ] Build and run instructions (README)
- [ ] Short writeup: approach, tradeoffs, known limitations
- [ ] `TODO` comments marking what would be done for production quality
- [ ] Zip of the project (including the writeup) for submission
- [ ] Preparation for a one-hour review meeting (walkthrough of code and approach)

## 3. API Contract

### `POST /detect`

**Input:** a document image as a multipart file upload (field name `file`).

**Output:**

```json
{
  "boxes": [
    { "bbox": [x1, y1, x2, y2], "is_checked": true },
    { "bbox": [x1, y1, x2, y2], "is_checked": false }
  ]
}
```

- `bbox` is pixel coordinates of the **top-left** (`x1, y1`) and **bottom-right** (`x2, y2`) corners, in the coordinate space of the uploaded image.

**Errors:**

| Case | Status |
|---|---|
| File can't be decoded as an image | 400 |
| Missing file field | 422 (FastAPI default) |
| Oversized file | 413 (TODO) |

### Optional extras

- `GET /health` for liveness checks
- `?debug=true` query param that returns an annotated image (green = unchecked, red = checked) to help tuning

## 4. Sample Data Characteristics

Four sample pages (real appraisal forms) came with the challenge. Things the solution must cope with:

| Characteristic | Detail |
|---|---|
| Mark styles | `X`, `✓`, odd or dotted box (page 4, Electricity row) |
| Noise | Strike-through lines, handwritten scribbles over boxes, a diagonal watermark (page 6), a rectangle drawn across text |
| Layout | Boxes sit next to or touch table rules |
| Scale | Different resolutions and aspect ratios, so size filters must be relative to image width |
| Density | Dozens of checkboxes per page, packed in rows |

## 5. Technical Approach

### Decision: classical CV first, learned model only where it fails

| Approach | Verdict |
|---|---|
| **OpenCV pipeline** | **Chosen as baseline.** No training data needed, fast, deterministic, easy to explain |
| YOLO fine-tune | Upgrade path. Needs labeled data (only 4 images available), so use public datasets or synthetic data |
| Hybrid (CV proposes, small CNN classifies) | Good second step if the fill threshold proves brittle |
| VLM (Claude, GPT, Florence-2) | Poor at precise bboxes on tiny objects, slow, costly, nondeterministic. At most a fallback for low-confidence crops |

### Detection pipeline

1. **Decode** the upload and convert to grayscale.
2. **Adaptive threshold** (inverted, so ink is white).
3. **Remove long lines:** morphological opening with long horizontal and vertical kernels, then subtract. Checkbox edges are short, so they survive; table rules don't.
4. **Connected components** with stats.
5. **Geometry filter:** width and height within a range scaled to image width; aspect ratio roughly 0.8 to 1.25.
6. **Ring check:** all four edges of the candidate mostly ink (rejects letters like O or D, and random blobs).
7. **Fill check:** ink density in the inset interior region (about 25% inset). Above a threshold means checked.
8. **NMS / deduplication** of overlapping detections.
9. **Serialize** to the response schema.

### Key tunables (store in one config module)

| Param | Starting value | Notes |
|---|---|---|
| Reference width | 1700 px | Scale factor `s = width / 1700` |
| Box size range | 8 to 40 px × `s` | Tune on samples |
| Aspect ratio range | 0.8 to 1.25 | |
| Adaptive threshold block / C | 31 / 15 | |
| Line-removal kernel | 40 px × `s` | |
| Ring edge ink ratio | ≥ 0.6 | |
| Fill threshold | > 0.10 | Learn from labeled data |

## 6. Suggested Project Structure

```
checkbox-detection/
├── app/
│   ├── main.py            # FastAPI app, /detect and /health
│   ├── detector.py        # CV pipeline (pure function: image in, boxes out)
│   ├── config.py          # Tunable parameters
│   └── schemas.py         # Pydantic response models
├── eval/
│   ├── labels/            # Hand-labeled ground truth (JSON per image)
│   ├── evaluate.py        # Precision / recall @ IoU 0.5, fill accuracy
│   └── visualize.py       # Draws predictions on images for debugging
├── samples/               # The 4 sample images
├── tests/
│   ├── test_detector.py
│   └── test_api.py
├── Dockerfile
├── requirements.txt
├── README.md              # Build, run, usage
└── WRITEUP.md             # Approach, tradeoffs, limitations
```

Keep `detector.py` free of FastAPI imports so it can be tested and evaluated on its own.

## 7. Build Plan (task checklist)

### Phase 1: Skeleton
- [ ] Create repo, virtualenv, `requirements.txt` (`fastapi`, `uvicorn`, `python-multipart`, `opencv-python-headless`, `numpy`, `pydantic`)
- [ ] FastAPI app with `POST /detect` returning a hardcoded response
- [ ] Image decode and 400 handling

### Phase 2: Baseline detector
- [ ] Threshold, line removal, connected components
- [ ] Geometry filter and ring check
- [ ] Fill check and `is_checked`
- [ ] NMS deduplication
- [ ] Wire into the endpoint

### Phase 3: Evaluation harness
- [ ] Hand-label the 4 sample pages (CVAT, Label Studio, or a quick script)
- [ ] `evaluate.py`: precision, recall, F1 at IoU 0.5; filled/unfilled accuracy on matched boxes
- [ ] `visualize.py`: overlay predictions and ground truth, save to `eval/out/`

### Phase 4: Tune and analyze
- [ ] Tune thresholds against the labeled pages
- [ ] Review failures and categorize them (missed, false positive, wrong state)
- [ ] Record before/after metrics

### Phase 5: Packaging
- [ ] Dockerfile (`python:3.11-slim`, headless OpenCV)
- [ ] README with build, run and curl example
- [ ] Basic tests (unit test for detector on a synthetic image; API test for 200 and 400)
- [ ] `TODO` comments for production work

### Phase 6: Writeup and submission
- [ ] `WRITEUP.md` (see section 9)
- [ ] Zip the project and submit
- [ ] Prepare for the review meeting (section 10)

### Stretch (only if time allows)
- [ ] Deskew step (Hough lines or `minAreaRect`)
- [ ] Small CNN classifier on proposed crops for filled/unfilled
- [ ] YOLO fine-tune on a public dataset plus synthetic data
- [ ] VLM fallback for low-confidence crops
- [ ] Confidence score per box in the response

## 8. Run Instructions (to put in README)

```bash
# Local
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Docker
docker build -t checkbox-detection .
docker run -p 8000:8000 checkbox-detection

# Try it
curl -X POST http://localhost:8000/detect \
  -F "file=@samples/page1.png"
```

## 9. Writeup Outline

1. **Problem and constraints:** what the sample documents look like.
2. **Approach:** the pipeline and why each step exists.
3. **Alternatives considered:** YOLO, hybrid CNN, VLM, and why they were not the first choice.
4. **Results:** precision, recall and fill accuracy on the 4 labeled pages, with before/after tuning.
5. **Tradeoffs:** accuracy vs. no-training-data simplicity; speed vs. robustness; relative size filters vs. fixed ones.
6. **Known limitations:**
   - Skewed or low-resolution scans
   - Boxes fused to table lines or each other
   - Scribbles, strike-throughs or watermarks crossing a box
   - Non-square or unusual boxes (dotted, rounded)
   - Light marks (faint ✓) falling below the fill threshold
   - Threshold tuned on only 4 pages, so likely overfit
7. **Production next steps:** deskew, learned detector, more labeled data, confidence scores, batching, auth and rate limiting, request size limits, structured logging, metrics.

## 10. Review Meeting Prep: "Why not X?"

| Question | Short answer |
|---|---|
| Why not a VLM? | Imprecise pixel bboxes on tiny objects, per-page cost and latency, nondeterministic output |
| Why not YOLO first? | No labeled data beyond 4 images; adds training and serving weight before a baseline exists |
| Why OpenCV? | Boxes are regular and high-contrast; fast, explainable, and gives a baseline to measure against |
| How do you know it works? | Eval harness with IoU-matched precision and recall on hand-labeled pages |
| What breaks it? | See limitations; show the failure overlays |
| How would you scale it? | Stateless service, horizontal scaling, queue for large batches, learned detector for robustness |
| How would you improve accuracy? | More labeled data, YOLO or CNN stage, deskew, per-box confidence with VLM fallback |

## 11. Definition of Done

- `POST /detect` returns valid JSON for all 4 sample images
- Service builds and runs from the README instructions on a clean machine
- Eval script reports numbers for the 4 labeled pages
- Writeup covers approach, tradeoffs, limitations
- Project zipped and submitted
