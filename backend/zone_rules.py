from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from backend.calibration import Calibration
from backend.detector import Detection
from backend.zones_store import Zone


@dataclass
class ZoneBreachEvent:
    """A person either approaching or entering a configured polygon zone.

    ``distance_*`` is signed: positive means outside the polygon, negative
    means inside.  The class name is retained for API compatibility with the
    earlier MVP, but the event now covers both approach and breach states.
    """

    zone: Zone
    person: Detection
    severity: str
    frame_timestamp: float
    rule_name: str = "zone_breach"
    inside: bool = True
    distance_px: float | None = None
    distance_m: float | None = None
    calibrated: bool = False
    confirmed: bool = False


def point_in_polygon(x: float, y: float, poly: list[list[float]] | list[tuple[float, float]]) -> bool:
    """Ray casting. ``poly`` must use the same coordinate space as ``x, y``."""
    if len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _point_segment_distance(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / length_sq
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy)


def signed_distance_to_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]] | list[list[float]],
) -> float:
    """Shortest signed distance to a polygon boundary.

    Positive values are outside, negative values are inside and zero means the
    point lies on the boundary.  The result uses the polygon's coordinate unit
    (pixels or metres).
    """
    if len(polygon) < 3:
        return float("inf")
    poly = [(float(p[0]), float(p[1])) for p in polygon]
    minimum = min(
        _point_segment_distance(point, poly[i], poly[(i + 1) % len(poly)])
        for i in range(len(poly))
    )
    if minimum <= 1e-9:
        return 0.0
    return -minimum if point_in_polygon(point[0], point[1], poly) else minimum


def _person_foot_point(det: Detection) -> tuple[float, float]:
    """Bottom-center of bbox — approximation of contact with ground."""
    x1, _y1, x2, y2 = det.box
    return ((x1 + x2) / 2.0, float(y2))


class ZoneBreachDetector:
    def evaluate(
        self,
        detections: list[Detection],
        zones: list[Zone],
        frame_w: int,
        frame_h: int,
        frame_timestamp: float = 0.0,
        calibration: Calibration | None = None,
    ) -> list[ZoneBreachEvent]:
        if not zones or frame_w <= 0 or frame_h <= 0:
            return []
        metric_calibration = (
            calibration
            if calibration is not None and calibration.is_compatible(frame_w, frame_h)
            else None
        )
        persons = [d for d in detections if d.category == "person"]
        events: list[ZoneBreachEvent] = []
        for person in persons:
            foot_px = _person_foot_point(person)
            for zone in zones:
                if not zone.active or len(zone.polygon) < 3:
                    continue

                polygon_px = [
                    (float(x) * frame_w, float(y) * frame_h)
                    for x, y in zone.polygon
                ]
                distance_px = signed_distance_to_polygon(foot_px, polygon_px)
                distance_m: float | None = None
                calibrated = metric_calibration is not None
                if metric_calibration is not None:
                    foot_m = metric_calibration.project(
                        foot_px[0], foot_px[1], frame_w, frame_h,
                    )
                    polygon_m = [
                        metric_calibration.project(x, y, frame_w, frame_h)
                        for x, y in polygon_px
                    ]
                    distance_m = signed_distance_to_polygon(foot_m, polygon_m)

                signed_distance = distance_m if distance_m is not None else distance_px
                inside = signed_distance <= 0.0
                if inside:
                    events.append(ZoneBreachEvent(
                        zone=zone,
                        person=person,
                        severity=zone.severity,
                        frame_timestamp=frame_timestamp,
                        rule_name="zone_breach",
                        inside=True,
                        distance_px=distance_px,
                        distance_m=distance_m,
                        calibrated=calibrated,
                    ))
                    continue

                warning_threshold = (
                    max(0.0, float(zone.warning_distance_m))
                    if distance_m is not None
                    else max(0.0, float(zone.warning_distance_px))
                )
                if signed_distance <= warning_threshold:
                    events.append(ZoneBreachEvent(
                        zone=zone,
                        person=person,
                        severity="WARNING",
                        frame_timestamp=frame_timestamp,
                        rule_name="zone_approach",
                        inside=False,
                        distance_px=distance_px,
                        distance_m=distance_m,
                        calibrated=calibrated,
                    ))
        return events


class ZoneTemporalFilter:
    """Confirm approach/breach only after N frames and apply cooldown.

    Approach and breach have independent streaks so crossing the boundary does
    not inherit stale confirmation state from the warning phase.
    """

    def __init__(self, required: int = 3, cooldown_sec: float = 8.0):
        self.required = required
        self.cooldown_sec = cooldown_sec
        self._streak: dict[str, int] = defaultdict(int)
        self._last_alert_time: dict[str, float] = {}

    @staticmethod
    def _key(event: ZoneBreachEvent) -> str:
        x1, y1, x2, y2 = event.person.box
        cell = ((x1 + x2) // 96, (y1 + y2) // 96)
        return f"{event.zone.id}_{event.rule_name}_{cell}"

    def update(self, events: list[ZoneBreachEvent], now: float) -> list[ZoneBreachEvent]:
        current_keys = set()
        confirmed: list[ZoneBreachEvent] = []
        for event in events:
            key = self._key(event)
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
