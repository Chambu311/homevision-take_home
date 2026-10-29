"""API tests for /health and /detect."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_health() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_detect_ok() -> None:
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (70, 70), (0, 0, 0), 2)
    res = client.post(
        "/detect",
        files={"file": ("box.png", _png_bytes(img), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert "boxes" in body
    assert isinstance(body["boxes"], list)


def test_detect_bad_file_returns_400() -> None:
    res = client.post(
        "/detect",
        files={"file": ("bad.bin", b"not-an-image", "application/octet-stream")},
    )
    assert res.status_code == 400


def test_detect_missing_file_returns_422() -> None:
    res = client.post("/detect")
    assert res.status_code == 422


def test_demo_page_served() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "HomeVision Checkbox Demo" in res.text
