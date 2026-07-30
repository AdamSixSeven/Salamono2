#!/usr/bin/env python3
"""Download a MediaPipe Pose Landmarker model."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import urllib.request

VARIANTS = {
    "lite": ("pose_landmarker_lite", 1_000_000),
    "full": ("pose_landmarker_full", 5_000_000),
    "heavy": ("pose_landmarker_heavy", 20_000_000),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="heavy")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_name, minimum_size = VARIANTS[args.variant]
    output = args.output or Path("models") / f"{model_name}.task"
    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"{model_name}/float16/latest/{model_name}.task"
    )

    if output.exists() and output.stat().st_size >= minimum_size and not args.force:
        print(f"Model already exists: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=output.parent) as tmp:
        temp_path = Path(tmp.name)
    try:
        print(f"Downloading {args.variant} pose model to {output}...")
        with urllib.request.urlopen(url, timeout=120) as response, temp_path.open("wb") as target:
            shutil.copyfileobj(response, target)
        if temp_path.stat().st_size < minimum_size:
            raise RuntimeError("Downloaded file is unexpectedly small")
        temp_path.replace(output)
        print(f"Saved {output} ({output.stat().st_size / 1_000_000:.1f} MB)")
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
