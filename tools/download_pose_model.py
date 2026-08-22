#!/usr/bin/env python3
"""Download the official MediaPipe Pose Landmarker Heavy task bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
DEFAULT_OUTPUT = Path("models/pose_landmarker_heavy.task")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output
    if output.exists() and not args.force:
        print(f"Model already exists: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=output.parent) as tmp:
        temp_path = Path(tmp.name)
    try:
        print(f"Downloading MediaPipe Pose Landmarker Heavy to {output} ...")
        with urllib.request.urlopen(MODEL_URL, timeout=120) as response, temp_path.open("wb") as f:
            shutil.copyfileobj(response, f)
        if temp_path.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded file is unexpectedly small")
        temp_path.replace(output)
        print(f"Saved {output} ({output.stat().st_size / 1_000_000:.1f} MB)")
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    main()
