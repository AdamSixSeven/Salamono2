from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math

from backend.calibration import Calibration
from backend.detector import Detection
from config import CONFIG, DangerConfig


@dataclass
class DangerEvent:
    rule_name: str
    person: Detection
    hazard: Detection
    distance_px: float
    overlap_iou: float
    severity: str  # "WARNING" or "DANGER"
    frame_timestamp: float
    distance_m: float | None = None
    calibrated: bool = False
    confirmed: bool = False


@dataclass
class DynamicSafetyZone:
    """A moving warning/danger polygon attached to a detected machine."""

    zone_id: str
    hazard: Detection
    severity: str
    polygon_px: list[tuple[float, float]]
    threshold_m: float | None
    threshold_px: float | None
    calibrated: bool


def bbox_iou(a: tuple, b: tuple) -> float:
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_min_distance(a: tuple, b: tuple) -> float:
    dx = max(0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0, max(a[1] - b[3], b[1] - a[3]))
    return (dx**2 + dy**2) ** 0.5


def _point_segment_distance(
    point: tuple[float, float],
    segment_a: tuple[float, float],
    segment_b: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = segment_a
    bx, by = segment_b
    vx, vy = bx - ax, by - ay
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / length_sq
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy)


def ground_distance_m(
    person: Detection,
    hazard: Detection,
    calibration: Calibration,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> float:
    """Approximate ground-plane distance from worker feet to machine footprint.

    The worker is represented by the bottom-centre of the person box.  The
    machine footprint is represented by the bottom edge of its box projected
    onto the calibrated plane.  This is a 2.5D homography measurement, not a
    full stereo reconstruction, but it is deterministic and useful for the MVP.
    """

    px = (person.box[0] + person.box[2]) / 2.0
    py = float(person.box[3])
    worker = calibration.project(px, py, frame_width, frame_height)
    left = calibration.project(
        float(hazard.box[0]), float(hazard.box[3]), frame_width, frame_height,
    )
    right = calibration.project(
        float(hazard.box[2]), float(hazard.box[3]), frame_width, frame_height,
    )
    return _point_segment_distance(worker, left, right)


class DangerDetector:
    def __init__(self, config: DangerConfig | None = None):
        self.cfg = config or CONFIG.danger

    @staticmethod
    def _hazards(detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if d.category in {"vehicle", "machine", "hazard"}]

    def evaluate(
        self,
        detections: list[Detection],
        frame_timestamp: float = 0.0,
        calibration: Calibration | None = None,
        frame_w: int | None = None,
        frame_h: int | None = None,
    ) -> list[DangerEvent]:
        metric_calibration = calibration
        if (
            metric_calibration is not None
            and frame_w is not None
            and frame_h is not None
            and not metric_calibration.is_compatible(frame_w, frame_h)
        ):
            metric_calibration = None
        persons = [d for d in detections if d.category == "person"]
        hazards = self._hazards(detections)
        events: list[DangerEvent] = []
        for person in persons:
            for hazard in hazards:
                iou = bbox_iou(person.box, hazard.box)
                dist_px = bbox_min_distance(person.box, hazard.box)
                dist_m = (
                    ground_distance_m(
                        person, hazard, metric_calibration, frame_w, frame_h,
                    )
                    if metric_calibration is not None
                    else None
                )

                if iou > self.cfg.overlap_iou:
                    events.append(DangerEvent(
                        rule_name="person_vehicle_overlap",
                        person=person,
                        hazard=hazard,
                        distance_px=0.0,
                        overlap_iou=iou,
                        severity="DANGER",
                        frame_timestamp=frame_timestamp,
                        distance_m=dist_m,
                        calibrated=metric_calibration is not None,
                    ))
                    continue

                if dist_m is not None:
                    if dist_m <= self.cfg.danger_distance_m:
                        rule_name, severity = "person_vehicle_danger_zone", "DANGER"
                    elif dist_m <= self.cfg.warning_distance_m:
                        rule_name, severity = "person_near_vehicle", "WARNING"
                    else:
                        continue
                else:
                    if dist_px <= self.cfg.danger_proximity_px:
                        rule_name, severity = "person_vehicle_danger_zone", "DANGER"
                    elif dist_px <= self.cfg.proximity_px:
                        rule_name, severity = "person_near_vehicle", "WARNING"
                    else:
                        continue

                events.append(DangerEvent(
                    rule_name=rule_name,
                    person=person,
                    hazard=hazard,
                    distance_px=dist_px,
                    overlap_iou=0.0,
                    severity=severity,
                    frame_timestamp=frame_timestamp,
                    distance_m=dist_m,
                    calibrated=metric_calibration is not None,
                ))
        return events

    def dynamic_zones(
        self,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
        calibration: Calibration | None = None,
    ) -> list[DynamicSafetyZone]:
        metric_calibration = (
            calibration
            if calibration is not None and calibration.is_compatible(frame_w, frame_h)
            else None
        )
        zones: list[DynamicSafetyZone] = []
        for index, hazard in enumerate(self._hazards(detections)):
            for severity, radius_m, radius_px in (
                ("WARNING", self.cfg.warning_distance_m, self.cfg.proximity_px),
                ("DANGER", self.cfg.danger_distance_m, self.cfg.danger_proximity_px),
            ):
                if metric_calibration is not None:
                    # Draw a metric capsule around the machine's projected
                    # ground-contact segment. This matches `ground_distance_m`
                    # and remains correct for long vehicles, unlike a circle
                    # centred only on the bbox midpoint.
                    left_m = metric_calibration.project(
                        float(hazard.box[0]), float(hazard.box[3]), frame_w, frame_h,
                    )
                    right_m = metric_calibration.project(
                        float(hazard.box[2]), float(hazard.box[3]), frame_w, frame_h,
                    )
                    dx = right_m[0] - left_m[0]
                    dy = right_m[1] - left_m[1]
                    theta = math.atan2(dy, dx)
                    count = max(12, self.cfg.dynamic_zone_points)
                    right_count = count // 2
                    left_count = count - right_count
                    metric_points: list[tuple[float, float]] = []
                    for i in range(right_count):
                        fraction = i / max(1, right_count - 1)
                        angle = theta - math.pi / 2.0 + math.pi * fraction
                        metric_points.append((
                            right_m[0] + radius_m * math.cos(angle),
                            right_m[1] + radius_m * math.sin(angle),
                        ))
                    for i in range(left_count):
                        fraction = i / max(1, left_count - 1)
                        angle = theta + math.pi / 2.0 + math.pi * fraction
                        metric_points.append((
                            left_m[0] + radius_m * math.cos(angle),
                            left_m[1] + radius_m * math.sin(angle),
                        ))
                    points = [
                        metric_calibration.unproject(
                            x_m, y_m, frame_w, frame_h,
                        )
                        for x_m, y_m in metric_points
                    ]
                    calibrated = True
                    threshold_m, threshold_px = radius_m, None
                else:
                    # Image-space fallback.  A padded rectangle is deliberately
                    # cheap and stable; after calibration it becomes a projected
                    # metric circle on the ground plane.
                    x1, y1, x2, y2 = hazard.box
                    pad = float(radius_px)
                    points = [
                        (max(0.0, x1 - pad), max(0.0, y1 - pad)),
                        (min(float(frame_w - 1), x2 + pad), max(0.0, y1 - pad)),
                        (min(float(frame_w - 1), x2 + pad), min(float(frame_h - 1), y2 + pad)),
                        (max(0.0, x1 - pad), min(float(frame_h - 1), y2 + pad)),
                    ]
                    calibrated = False
                    threshold_m, threshold_px = None, float(radius_px)

                zones.append(DynamicSafetyZone(
                    zone_id=f"machine-{index}-{severity.lower()}",
                    hazard=hazard,
                    severity=severity,
                    polygon_px=points,
                    threshold_m=threshold_m,
                    threshold_px=threshold_px,
                    calibrated=calibrated,
                ))
        return zones


class TemporalFilter:
    def __init__(self, required: int = 3, cooldown_sec: float = 10.0):
        self.required = required
        self.cooldown_sec = cooldown_sec
        self._streak: dict[str, int] = defaultdict(int)
        self._last_alert_time: dict[str, float] = {}

    @staticmethod
    def _pair_key(event: DangerEvent) -> str:
        def center(box):
            return ((box[0] + box[2]) // 64, (box[1] + box[3]) // 64)
        pc = center(event.person.box)
        vc = center(event.hazard.box)
        return f"{event.rule_name}_{pc}_{vc}"

    def update(self, events: list[DangerEvent], now: float) -> list[DangerEvent]:
        current_keys = set()
        confirmed = []
        for event in events:
            key = self._pair_key(event)
            current_keys.add(key)
            self._streak[key] += 1
            if self._streak[key] >= self.required:
                last = self._last_alert_time.get(key, float("-inf"))
                if now - last >= self.cooldown_sec:
                    event.confirmed = True
                    confirmed.append(event)
                    self._last_alert_time[key] = now

        dead = [key for key in self._streak if key not in current_keys]
        for key in dead:
            del self._streak[key]
        return confirmed
