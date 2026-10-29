"""HomeVision Checkbox Detection API.

TODO(production):
- Enforce max upload size and return 413 for oversized files.
- Add auth, rate limiting, request IDs, and structured logging/metrics.
- Optional async offload of detection for large batches.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from app.detector import annotate_detections, decode_image, detect_checkboxes
from app.schemas import CheckboxBox, DetectResponse

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="HomeVision Checkbox Detection",
    version="0.1.0",
    description="Detect and classify checkboxes in appraisal document images.",
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_home() -> FileResponse:
    return FileResponse(STATIC_DIR / "demo.html")


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "demo.html")


@app.get("/demo/samples", tags=["demo"])
def list_samples() -> dict:
    samples = []
    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        samples.append({"name": path.name, "width": w, "height": h})
    return {"samples": samples}


def _sample_path(name: str) -> Path:
    # Prevent path traversal; only allow files that live directly in samples/
    candidate = (SAMPLES_DIR / Path(name).name).resolve()
    if candidate.parent != SAMPLES_DIR.resolve() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Sample not found")
    return candidate


@app.get("/demo/detect/{name}", tags=["demo"])
def demo_detect_stats(name: str) -> dict:
    path = _sample_path(name)
    image = decode_image(path.read_bytes())
    if image is None:
        raise HTTPException(status_code=400, detail="Sample cannot be decoded as an image")
    detections = detect_checkboxes(image)
    checked = sum(1 for d in detections if d.is_checked)
    return {
        "name": name,
        "total": len(detections),
        "checked": checked,
        "unchecked": len(detections) - checked,
        "boxes": [
            {"bbox": list(d.bbox), "is_checked": d.is_checked} for d in detections
        ],
    }


@app.get("/demo/annotate/{name}", tags=["demo"])
def demo_annotate(name: str) -> Response:
    path = _sample_path(name)
    image = decode_image(path.read_bytes())
    if image is None:
        raise HTTPException(status_code=400, detail="Sample cannot be decoded as an image")
    detections = detect_checkboxes(image)
    annotated = annotate_detections(image, detections)
    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.post(
    "/detect",
    response_model=None,
    tags=["detection"],
    summary="Detect checkboxes in an uploaded document image",
)
async def detect(
    file: UploadFile = File(..., description="Document image (PNG/JPEG)"),
    debug: bool = Query(
        False,
        description="If true, return an annotated PNG (green=unchecked, red=checked)",
    ),
) -> DetectResponse | Response:
    # TODO(production): reject when Content-Length / bytes exceed a configured max (413)
    data = await file.read()
    image = decode_image(data)
    if image is None:
        raise HTTPException(status_code=400, detail="File cannot be decoded as an image")

    detections = detect_checkboxes(image)

    if debug:
        annotated = annotate_detections(image, detections)
        ok, encoded = cv2.imencode(".png", annotated)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode debug image")
        return Response(content=encoded.tobytes(), media_type="image/png")

    return DetectResponse(
        boxes=[
            CheckboxBox(bbox=list(d.bbox), is_checked=d.is_checked) for d in detections
        ]
    )
