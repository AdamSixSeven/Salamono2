from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes import workers
from backend.worker_tags import WORKER_TAG_DICTIONARY_ID, worker_marker_id


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(workers.router, prefix="/api")
    return TestClient(app)


def test_worker_tag_png_is_printable_and_decodable():
    response = _client().get("/api/worker-tag.png", params={"worker_id": "W-001"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.headers["x-worker-tag-type"] == "aruco-4x4"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")

    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    dictionary = cv2.aruco.getPredefinedDictionary(WORKER_TAG_DICTIONARY_ID)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
    assert ids is not None
    assert worker_marker_id("W-001") in ids.flatten().tolist()


def test_worker_tag_png_download_header_and_safe_filename():
    response = _client().get(
        "/api/worker-tag.png",
        params={"worker_id": "W 001/A", "download": "true"},
    )
    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert 'filename="worker-W_001_A.png"' in disposition


def test_legacy_worker_qr_routes_are_removed():
    assert _client().get("/api/worker-qr", params={"worker_id": "W-001"}).status_code == 404
    assert _client().get("/api/worker-qr.png", params={"worker_id": "W-001"}).status_code == 404
