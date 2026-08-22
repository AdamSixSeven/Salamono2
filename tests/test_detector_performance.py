import sys
import types

import numpy as np
import pytest

from config import YOLOConfig
from backend.detector import Detector


class _FakeModel:
    names = {0: "person"}

    def __init__(self):
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append((frame, kwargs))
        return [types.SimpleNamespace(boxes=[])]


@pytest.mark.parametrize(
    ("device", "precision", "expected_quantize"),
    [
        ("0", "fp16", 16),
        ("cuda:0", "fp16", 16),
        ("0", "fp32", 32),
        ("cpu", "fp16", 32),
        ("mps", "fp16", 32),
        ("mps:0", "fp16", 32),
    ],
)
def test_detector_uses_current_quantize_api_and_safe_precision(
    monkeypatch, device, precision, expected_quantize
):
    model = _FakeModel()
    fake_ultralytics = types.SimpleNamespace(YOLO=lambda _name: model)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    detector = Detector(
        config=YOLOConfig(
            model_name="fake.pt",
            device=device,
            precision=precision,
            img_size=512,
            person_class_ids=[0],
            hazard_class_ids=[],
        ),
        categories={"person": [0]},
    )

    assert detector.detect(np.zeros((16, 16, 3), dtype=np.uint8)) == []
    kwargs = model.calls[0][1]
    assert kwargs["quantize"] == expected_quantize
    assert "half" not in kwargs
    assert kwargs["imgsz"] == 512
