from __future__ import annotations

import re
import zlib

import cv2
import numpy as np

# Simpler and more robust than a dense QR for long-range viewing. 4x4 keeps
# the visual complexity low while DICT_4X4_1000 provides enough unique IDs for
# worker badges such as W-001..W-999.
WORKER_TAG_DICTIONARY_ID = getattr(cv2.aruco, "DICT_4X4_1000", cv2.aruco.DICT_4X4_250)
WORKER_TAG_SIZE = 1000 if WORKER_TAG_DICTIONARY_ID == getattr(cv2.aruco, "DICT_4X4_1000", -1) else 250
DEFAULT_WORKER_ID_PREFIX = "W-"


def worker_tag_dictionary():
    return cv2.aruco.getPredefinedDictionary(WORKER_TAG_DICTIONARY_ID)


def worker_marker_id(worker_id: str, dictionary_size: int = WORKER_TAG_SIZE) -> int:
    value = str(worker_id or "").strip()
    if not value:
        raise ValueError("worker_id is required")
    match = re.search(r"(\d+)$", value)
    if match:
        numeric = int(match.group(1))
        if 0 <= numeric < dictionary_size:
            return numeric
    # Fallback for non-numeric IDs. Stable hash, low collision risk in small teams.
    return int(zlib.crc32(value.encode("utf-8")) % max(1, dictionary_size))


def worker_id_from_marker_id(marker_id: int) -> str:
    marker_id = int(marker_id)
    if marker_id < 0:
        marker_id = 0
    return f"{DEFAULT_WORKER_ID_PREFIX}{marker_id:03d}"


def render_worker_marker(marker_id: int, side_pixels: int = 1000, border_bits: int = 1) -> np.ndarray:
    dictionary = worker_tag_dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, int(marker_id), int(side_pixels), borderBits=int(border_bits))
    img = np.zeros((int(side_pixels), int(side_pixels)), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, int(marker_id), int(side_pixels), img, int(border_bits))
    return img
