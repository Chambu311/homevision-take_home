# HomeVision — Checkbox Detection (Take-Home)

Backend take-home assignment for **HomeVision**: an HTTP service that finds checkboxes on appraisal document images and classifies each as checked or unchecked.

**For reviewers:** start with [Quick start](#quick-start-5-minutes), try the [visual demo](#visual-demo), skim [WRITEUP.md](WRITEUP.md) for approach and tradeoffs, then walk [Project layout](#project-layout) (`app/detector.py` is the core).

Original prompt: [CHECKBOX_DETECTION_SPEC.md](CHECKBOX_DETECTION_SPEC.md)

---

## What this delivers

| Deliverable | Location |
|---|---|
| `POST /detect` + `GET /health` | [app/main.py](app/main.py) |
| Classical OpenCV detector | [app/detector.py](app/detector.py) |
| Tunable parameters | [app/config.py](app/config.py) |
| Sample pages (4) | [samples/](samples/) |
| Approach, tradeoffs, limitations | [WRITEUP.md](WRITEUP.md) |
| Eval harness | [eval/evaluate.py](eval/evaluate.py), [eval/visualize.py](eval/visualize.py) |
| Tests | [tests/](tests/) |
| Docker / Compose | [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml) |
| Production `TODO`s | Comments in `app/main.py`, `app/detector.py` |

---

## Quick start (5 minutes)

**Requirements:** Python 3.11+ (3.13 works locally), or Docker.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open:

| URL | What to try |
|---|---|
| http://127.0.0.1:8000/ | **Demo UI** — click each sample; red = checked, green = unchecked |
| http://127.0.0.1:8000/docs | **Swagger** — try `POST /detect` with a file upload |
| http://127.0.0.1:8000/health | Liveness (`{"status":"ok"}`) |

**Smoke test from the terminal:**

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/detect -F "file=@samples/0.png" | head -c 400
pytest -q
```

---

## API usage

Clients send images over **HTTP** as **multipart form uploads** (field name **`file`**). PNG and JPEG are supported.

### `POST /detect`

**Request:** `multipart/form-data`, part `file` = image bytes.

**Response (JSON):**

```json
{
  "boxes": [
    { "bbox": [x1, y1, x2, y2], "is_checked": true },
    { "bbox": [x1, y1, x2, y2], "is_checked": false }
  ]
}
```

- `bbox`: pixel coordinates of top-left `(x1, y1)` and bottom-right `(x2, y2)` in the uploaded image.
- `is_checked`: `true` if the box appears marked (X, ✓, fill), else `false`.

**Query:** `?debug=true` returns an annotated **PNG** instead of JSON (green = unchecked, red = checked).

**Errors:**

| Case | HTTP |
|---|---|
| Missing `file` | 422 |
| Not decodable as an image | 400 |
| Oversized upload | 413 (not implemented yet — see `TODO` in code) |

### Examples

```bash
# JSON result
curl -X POST http://127.0.0.1:8000/detect \
  -F "file=@samples/1.jpg"

# Annotated image
curl -X POST "http://127.0.0.1:8000/detect?debug=true" \
  -F "file=@samples/2.png" \
  --output annotated.png
```

```javascript
// Browser / Node (fetch)
const form = new FormData();
form.append("file", imageFile);
const res = await fetch("http://127.0.0.1:8000/detect", { method: "POST", body: form });
const { boxes } = await res.json();
```

There is **no** URL-fetch endpoint; the server only reads bytes from the upload body.

### `GET /health`

Returns `{"status":"ok"}` for load balancers and Docker healthchecks.

---

## Visual demo

The page at `/` is optional sugar on top of the same API:

1. Lists the four files in `samples/`.
2. Runs detection and shows counts (checked vs unchecked).
3. Displays an annotated preview.
4. Lets you upload any local image.

Demo-only routes (also usable for scripting): `GET /demo/samples`, `GET /demo/detect/{name}`, `GET /demo/annotate/{name}`.

---

## Docker (production / Dokploy)

Local development uses the [Quick start](#quick-start-5-minutes) (`uvicorn`). Compose is for deploying behind Traefik/Dokploy.

```bash
docker compose up -d --build
```

`docker-compose.yml` joins **`dokploy-network`** and does **not** bind a host port. In Dokploy Domains: service **`api`**, container port **`8000`**, then **Redeploy** after saving the domain. If you get Traefik `404 page not found`, check the container is **healthy**.

---

## Evaluation and tests

```bash
# Unit + API tests
pytest -q

# Draw predictions on samples → eval/out/
python -m eval.visualize

# Precision / recall @ IoU 0.5 (labels bootstrapped from detector — see WRITEUP)
python -m eval.evaluate
```

---

## Project layout

```
app/
  main.py           # FastAPI: /detect, /health, demo routes
  detector.py       # OpenCV pipeline (testable without FastAPI)
  config.py         # Thresholds (reference width 1700px, scaled per image)
  schemas.py        # Pydantic models
  static/demo.html  # Visual tester
samples/            # 0.png, 1.jpg, 2.png, 3.png — appraisal pages
eval/
  evaluate.py       # Metrics script
  visualize.py      # Overlay script
  labels/           # JSON ground truth (bootstrap baseline)
tests/
WRITEUP.md          # Approach, alternatives, limitations (review meeting)
CHECKBOX_DETECTION_SPEC.md
Dockerfile
docker-compose.yml
```

**Detection pipeline (high level):** adaptive threshold → remove long table lines → connected components → square-ring test (rejects letters) → hole proposals (table-fused boxes) → fill classification → NMS. Details in [WRITEUP.md](WRITEUP.md).

---

## Review meeting pointers

- **Why classical CV first?** No labeled training set beyond four pages; fast, explainable baseline.
- **Where to look in code:** `detect_checkboxes()` in [app/detector.py](app/detector.py); tunables in [app/config.py](app/config.py).
- **Known misses:** dotted boxes, heavy scribbles over a box — listed in WRITEUP.
- **Production gaps:** upload size limits, auth, deskew, learned classifier — marked with `TODO(production)` in source.

---

## License / attribution

Take-home submission for HomeVision. Sample images are appraisal-style forms provided for the exercise.
