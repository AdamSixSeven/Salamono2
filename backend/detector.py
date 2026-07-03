from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from config import CONFIG, YOLOConfig

CLASSES_OF_INTEREST = {
    "person": [0],
    "vehicle": [2, 5, 7],  # car, bus, truck
}
ALL_CLASSES = [c for group in CLASSES_OF_INTEREST.values() for c in group]

_CATEGORY_LOOKUP = {}
for cat, ids in CLASSES_OF_INTEREST.items():
    for cid in ids:
        _CATEGORY_LOOKUP[cid] = cat


@dataclass
class Detection:
    class_id: int
    class_name: str
    category: str  # "person" or "vehicle"
    box: tuple  # (x1, y1, x2, y2)
    confidence: float
    track_id: int | None = None


class Detector:
    def __init__(self, config: YOLOConfig | None = None):
        cfg = config or CONFIG.yolo
        self.model = YOLO(cfg.model_name)
        self.conf = cfg.confidence_threshold
        self.iou = cfg.iou_threshold
        self.device = cfg.device
        self.img_size = cfg.img_size

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.img_size,
            classes=ALL_CLASSES,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=self.model.names[cls_id],
                    category=_CATEGORY_LOOKUP.get(cls_id, "unknown"),
                    box=tuple(int(v) for v in box.xyxy[0].tolist()),
                    confidence=float(box.conf[0]),
                ))
        return detections
