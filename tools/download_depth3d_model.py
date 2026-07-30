#!/usr/bin/env python3
"""Pre-download the optional monocular metric-depth model into HF cache."""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Depth Anything V2 metric model used by the depth module.")
    parser.add_argument(
        "--model",
        default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        help="Hugging Face model id; the default is Indoor Small. Use Metric-Outdoor-Small for outdoor construction-site scenes.",
    )
    args = parser.parse_args()
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except Exception as exc:
        raise SystemExit(
            "Install optional dependencies first: "
            "python -m pip install -r requirements-depth3d.txt"
        ) from exc

    print(f"Downloading {args.model}...")
    AutoImageProcessor.from_pretrained(args.model)
    AutoModelForDepthEstimation.from_pretrained(args.model)
    print("Model is ready in the Hugging Face cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
