"""Bounded JPEG encoder used to overlap demo preview transport with GPU work."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class EncodedPreview:
    jpeg_bytes: bytes
    encode_ms: float


class PreviewJpegEncoder:
    """One reusable worker; backend-decoded video has at most one frame ahead."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="preview-jpeg",
        )

    @staticmethod
    def _encode(frame: np.ndarray, quality: int) -> EncodedPreview:
        started = time.perf_counter()
        encoded, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not encoded:
            raise ValueError("decoded frame could not be encoded for preview")
        return EncodedPreview(buffer.tobytes(), elapsed_ms)

    def submit(self, frame: np.ndarray, quality: int) -> Future[EncodedPreview]:
        # The decoded job owns ``frame`` until publication, and all concurrent
        # consumers are read-only. Avoiding another full-frame copy is material
        return self._executor.submit(self._encode, frame, int(quality))

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


__all__ = ["EncodedPreview", "PreviewJpegEncoder"]
