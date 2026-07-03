import os
from datetime import datetime, timezone

import cv2
import numpy as np

from config import CONFIG


class FrameStore:
    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or CONFIG.flagged_frames_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, frame: np.ndarray, alert_id: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{alert_id}.jpg"
        filepath = os.path.join(self.base_dir, filename)
        cv2.imwrite(filepath, frame)
        return f"/static/flagged/{filename}"

    def list_recent(self, limit: int = 50) -> list[dict]:
        if not os.path.exists(self.base_dir):
            return []
        files = sorted(os.listdir(self.base_dir), reverse=True)[:limit]
        return [{"filename": f, "url": f"/static/flagged/{f}"} for f in files]
