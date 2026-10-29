# HomeVision Checkbox Detection

HTTP API that finds checkboxes in appraisal document images and classifies each as checked or unchecked.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8001
```

| Endpoint | Purpose |
|---|---|
| http://127.0.0.1:8001/ | **HTML demo** — click each sample, see red/green overlays |
| http://127.0.0.1:8001/docs | OpenAPI docs |
| `GET /health` | Liveness |
| `POST /detect` | JSON boxes |
| `POST /detect?debug=true` | Annotated PNG |

### Curl

```bash
curl -X POST http://127.0.0.1:8001/detect \
  -F "file=@samples/0.png"

# Annotated preview
curl -X POST "http://127.0.0.1:8001/detect?debug=true" \
  -F "file=@samples/0.png" \
  --output debug.png
```

### Docker

```bash
docker build -t checkbox-detection .
docker run -p 8000:8000 checkbox-detection
```

### Docker Compose (VPS)

On the server, clone the repo (or copy the project), then:

```bash
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

Change the host port with `API_PORT=8080 docker compose up -d --build`.

| URL | Purpose |
|---|---|
| `http://YOUR_VPS_IP:8000/` | HTML demo |
| `http://YOUR_VPS_IP:8000/docs` | OpenAPI |
| `http://YOUR_VPS_IP:8000/detect` | Detection API |

Logs: `docker compose logs -f api`  
Restart: `docker compose restart api`  
Stop: `docker compose down`

Put nginx or Caddy in front for HTTPS and request size limits in production.

## Visual demo

Open `/` after starting the server. The sidebar lists `samples/0.png` … `3.png`. Selecting one:

1. Runs detection
2. Shows **checked** (red) vs **unchecked** (green) counts
3. Renders an annotated preview of the page

You can also upload any local image from the same page.

## Project layout

```
app/
  main.py          # FastAPI: /detect, /health, HTML demo routes
  detector.py      # OpenCV pipeline (no FastAPI imports)
  config.py        # Tunable thresholds
  schemas.py       # Response models
  static/demo.html # Visual test UI
samples/           # 0.png, 1.jpg, 2.png, 3.png
eval/              # evaluate.py, visualize.py, labels/
tests/
```

## Eval

```bash
python -m eval.visualize
python -m eval.evaluate
pytest
```

## Writeup

See [WRITEUP.md](WRITEUP.md) for approach, tradeoffs, and limitations.
# homevision-take_home
