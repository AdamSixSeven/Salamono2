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
    ("device", "expected_half"),
    [
        ("0", True),
        ("cuda:0", True),
        ("cpu", False),
        ("mps", False),
        ("mps:0", False),
    ],
)
def test_detector_enables_half_only_for_cuda_or_accelerator_devices(
    monkeypatch, device, expected_half
):
    model = _FakeModel()
    fake_ultralytics = types.SimpleNamespace(YOLO=lambda _name: model)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    detector = Detector(
        config=YOLOConfig(
            model_name="fake.pt",
            device=device,
            img_size=512,
            person_class_ids=[0],
            hazard_class_ids=[],
        ),
        categories={"person": [0]},
    )

    assert detector.detect(np.zeros((16, 16, 3), dtype=np.uint8)) == []
    kwargs = model.calls[0][1]
    assert kwargs["half"] is expected_half
    assert kwargs["imgsz"] == 512
