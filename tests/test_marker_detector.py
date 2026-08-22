import cv2
import numpy as np
import pytest

from backend.marker_detector import DEFAULT_DICT, MarkerDetector


def _marker_image(marker_ids, size_px=200, margin_px=40):
    """Render a big BGR frame with marker_ids arranged in a 2x2 grid."""
    d = cv2.aruco.getPredefinedDictionary(DEFAULT_DICT)
    cols = 2
    rows = 2
    cell = size_px + margin_px * 2
    frame = np.full((rows * cell, cols * cell, 3), 255, dtype=np.uint8)
    for i, mid in enumerate(marker_ids[:4]):
        r, c = divmod(i, cols)
        # Prefer the OpenCV 4.7+ name; fall back to legacy for older builds.
        if hasattr(cv2.aruco, "generateImageMarker"):
            img = cv2.aruco.generateImageMarker(d, mid, size_px)
        else:  # pragma: no cover — legacy path
            img = cv2.aruco.drawMarker(d, mid, size_px)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        y0 = r * cell + margin_px
        x0 = c * cell + margin_px
        frame[y0:y0 + size_px, x0:x0 + size_px] = img_bgr
    return frame


def test_detect_finds_all_markers():
    ids = [3, 7, 11, 23]
    frame = _marker_image(ids)
    detector = MarkerDetector()
    detections = detector.detect(frame)
    got_ids = sorted(d.marker_id for d in detections)
    assert got_ids == sorted(ids)


def test_detect_returns_reasonable_centers():
    frame = _marker_image([3, 7, 11, 23])
    detector = MarkerDetector()
    detections = detector.detect(frame)
    assert len(detections) == 4
    h, w = frame.shape[:2]
    for d in detections:
        cx, cy = d.center
        assert 0 <= cx <= w
        assert 0 <= cy <= h
        assert len(d.corners) == 4


def test_empty_frame_returns_no_detections():
    frame = np.full((400, 400, 3), 128, dtype=np.uint8)
    detector = MarkerDetector()
    assert detector.detect(frame) == []
