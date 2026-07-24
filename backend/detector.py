from dataclasses import dataclass

import numpy as np
from config import CONFIG, YOLOConfig

SITE_CATEGORIES: dict[str, list[int]] = {
    "person": [0],
    "vehicle": [2, 5, 7],  # car, bus, truck (COCO)
}

PPE_CATEGORIES: dict[str, list[int]] = {
    "person": [5],
    "hardhat": [0],
    "no_hardhat": [2],
    "vest": [7],
    "no_vest": [4],
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    category: str
    box: tuple  # (x1, y1, x2, y2)
    confidence: float
    track_id: int | None = None


class Detector:
    def __init__(
        self,
        model_name: str | None = None,
        categories: dict[str, list[int]] | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        device: str | None = None,
        img_size: int | None = None,
        config: YOLOConfig | None = None,
    ):
        cfg = config or CONFIG.yolo
        # Lazy import keeps lightweight rule/unit tests independent from the
        # heavyweight inference runtime. Production still fails clearly when
        # Detector is actually instantiated without ultralytics installed.
        from ultralytics import YOLO
        self.model = YOLO(model_name or cfg.model_name)
        self.categories = categories or SITE_CATEGORIES
        self._class_to_category: dict[int, str] = {}
        for cat, ids in self.categories.items():
            for cid in ids:
                self._class_to_category[cid] = cat
        self._class_filter = list(self._class_to_category.keys())
        self.conf = confidence if confidence is not None else cfg.confidence_threshold
        self.iou = iou if iou is not None else cfg.iou_threshold
        self.device = device or cfg.device
        self.img_size = img_size or cfg.img_size

    def detect(self, frame: np.ndarray) -> list[Detection]:
        device_name = str(self.device).strip().lower()
        use_half = device_name not in {"cpu", "mps"} and not device_name.startswith("mps:")
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.img_size,
            classes=self._class_filter,
            # FP16 substantially reduces CUDA/TensorRT inference cost.  It is
            # deliberately disabled on CPU and Apple MPS where half precision
            # is unsupported or can be slower/less stable.
            half=use_half,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=self.model.names[cls_id],
                    category=self._class_to_category.get(cls_id, "unknown"),
                    box=tuple(int(v) for v in box.xyxy[0].tolist()),
                    confidence=float(box.conf[0]),
                ))
        return detections
