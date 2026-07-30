#!/usr/bin/env python3
"""Download standard runtime models."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import urllib.request

POSE_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"

MODELS = {
    "pose-lite": (
        f"{POSE_BASE}/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        Path("models/pose_landmarker_lite.task"),
        1_000_000,
    ),
    "pose-full": (
        f"{POSE_BASE}/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        Path("models/pose_landmarker_full.task"),
        5_000_000,
    ),
    "pose-heavy": (
        f"{POSE_BASE}/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        Path("models/pose_landmarker_heavy.task"),
        20_000_000,
    ),
    "ppe": (
        "https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt",
        Path("ppe.pt"),
        1_000_000,
    ),
}

DEFAULT_MODELS = ("pose-heavy", "ppe")


def download(name: str, force: bool = False) -> Path:
    url, target, minimum_size = MODELS[name]
    if target.exists() and target.stat().st_size >= minimum_size and not force:
        print(f"[ok] {name}: {target} ({target.stat().st_size / 1_000_000:.1f} MB)")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {name}: {url}")
    with tempfile.NamedTemporaryFile(delete=False, dir=str(target.parent)) as tmp:
        temp_path = Path(tmp.name)

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Perimetr/2.2"})
        with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        size = temp_path.stat().st_size
        if size < minimum_size:
            raise RuntimeError(f"Downloaded file is unexpectedly small: {size} B")
        temp_path.replace(target)
        print(f"[saved] {target} ({size / 1_000_000:.1f} MB)")
        return target
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", choices=sorted(MODELS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = args.models or DEFAULT_MODELS
    for name in selected:
        download(name, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
