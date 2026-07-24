from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes import workers


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(workers.router, prefix="/api")
    return TestClient(app)


def test_worker_qr_png_is_printable_and_decodable():
    response = _client().get("/api/worker-qr.png", params={"worker_id": "W-001"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")

    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    decoded, _points, _straight = cv2.QRCodeDetector().detectAndDecode(image)
    assert decoded == "worker:W-001"


def test_worker_qr_png_download_header_and_safe_filename():
    response = _client().get(
        "/api/worker-qr.png",
        params={"worker_id": "W 001/A", "download": "true"},
    )
    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert 'filename="worker-W_001_A.png"' in disposition


def test_legacy_svg_worker_qr_still_works():
    response = _client().get("/api/worker-qr", params={"worker_id": "W-001"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
