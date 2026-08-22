#!/usr/bin/env python3
"""Load the bundled scene detector and verify the runtime class contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG  # noqa: E402

EXPECTED_NAMES = {
    0: "person",
    1: "car",
    2: "bus",
    3: "truck",
    4: "excavator",
    5: "bulldozer",
    6: "grader",
    7: "loader",
    8: "mobile_crane",
    9: "road_roller",
}


def normalized_names(raw_names) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    return {index: str(value) for index, value in enumerate(raw_names)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=CONFIG.yolo.model_name,
        help="Checkpoint path; defaults to YOLO_MODEL/runtime configuration.",
    )
    parser.add_argument(
        "--device",
        default=CONFIG.yolo.device,
        help="Ultralytics device argument, e.g. auto, cpu, 0 or cuda:0.",
    )
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.is_file():
        print(f"ERROR: model file not found: {model_path}")
        return 2

    model = YOLO(str(model_path))
    names = normalized_names(model.names)
    if names != EXPECTED_NAMES:
        print("ERROR: checkpoint class map does not match scene-v3")
        print("expected:", EXPECTED_NAMES)
        print("actual:  ", names)
        return 3

    person_ids = list(CONFIG.yolo.person_class_ids)
    hazard_ids = list(CONFIG.yolo.hazard_class_ids)
    if person_ids != [0] or hazard_ids != list(range(1, 10)):
        print("ERROR: runtime class mapping does not match scene-v3")
        print("person IDs:", person_ids)
        print("hazard IDs:", hazard_ids)
        return 4

    print("Model:", model_path)
    print("Classes:", names)
    print("Person IDs:", person_ids)
    print("Hazard IDs:", hazard_ids)

    if not args.skip_inference:
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            device=args.device,
            classes=person_ids + hazard_ids,
            verbose=False,
        )
        print("Inference smoke test: OK; detections:", sum(len(r.boxes) for r in results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
