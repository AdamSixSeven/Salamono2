#!/usr/bin/env python3
"""Download optional runtime models needed by the complete MVP.

The application still starts when an optional model is missing, but this
helper makes the local setup equivalent to the Docker image:
- MediaPipe Pose Landmarker Lite for posture/fall heuristics;
- a public PPE checkpoint for the checkpoint demo.

Use a custom trained checkpoint for production construction equipment/PPE.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import tempfile
import urllib.request

MODELS = {
    "pose": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        Path("models/pose_landmarker_lite.task"),
        1_000_000,
    ),
    "ppe": (
        "https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt",
        Path("ppe.pt"),
        1_000_000,
    ),
}


def download(name: str, force: bool = False) -> Path:
    url, target, minimum_size = MODELS[name]
    if target.exists() and target.stat().st_size >= minimum_size and not force:
        print(f"[ok] {name}: {target} ({target.stat().st_size / 1_000_000:.1f} MB)")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {name}: {url}")
    with tempfile.NamedTemporaryFile(delete=False, dir=str(target.parent)) as tmp:
        tmp_path = Path(tmp.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Perimetr-MVP/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        size = tmp_path.stat().st_size
        if size < minimum_size:
            raise RuntimeError(
                f"Pobrany plik {name} ma tylko {size} B; prawdopodobnie pobrano stronę błędu."
            )
        tmp_path.replace(target)
        print(f"[saved] {target} ({size / 1_000_000:.1f} MB)")
        return target
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        choices=sorted(MODELS),
        default=list(MODELS),
        help="Modele do pobrania; bez argumentów pobiera komplet MVP.",
    )
    parser.add_argument("--force", action="store_true", help="Pobierz ponownie istniejące pliki")
    args = parser.parse_args()

    selected_models = args.models or list(MODELS)

    for name in selected_models:
        download(name, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
