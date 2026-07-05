from collections import defaultdict
from dataclasses import dataclass

from backend.detector import Detection
from backend.zones_store import Zone


@dataclass
class ZoneBreachEvent:
    zone: Zone
    person: Detection
    severity: str
    frame_timestamp: float
    confirmed: bool = False


def point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    """Ray casting. `poly` in same coordinate space as (x, y)."""
    if len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def _person_foot_point(det: Detection) -> tuple[float, float]:
    """Bottom-center of bbox — closer to where feet touch ground."""
    x1, y1, x2, y2 = det.box
    return ((x1 + x2) / 2.0, float(y2))


class ZoneBreachDetector:
    def evaluate(
        self,
        detections: list[Detection],
        zones: list[Zone],
        frame_w: int,
        frame_h: int,
        frame_timestamp: float = 0.0,
    ) -> list[ZoneBreachEvent]:
        if not zones or frame_w <= 0 or frame_h <= 0:
            return []
        persons = [d for d in detections if d.category == "person"]
        events: list[ZoneBreachEvent] = []
        for p in persons:
            fx, fy = _person_foot_point(p)
            nx, ny = fx / frame_w, fy / frame_h
            for z in zones:
                if not z.active or len(z.polygon) < 3:
                    continue
                if point_in_polygon(nx, ny, z.polygon):
                    events.append(ZoneBreachEvent(
                        zone=z,
                        person=p,
                        severity=z.severity,
                        frame_timestamp=frame_timestamp,
                    ))
        return events


class ZoneTemporalFilter:
    """Same shape as danger_rules.TemporalFilter — confirm after N frames,
    cooldown between alerts per (zone_id, person-cell)."""

    def __init__(self, required: int = 3, cooldown_sec: float = 8.0):
        self.required = required
        self.cooldown_sec = cooldown_sec
        self._streak: dict[str, int] = defaultdict(int)
        self._last_alert_time: dict[str, float] = {}

    @staticmethod
    def _key(event: ZoneBreachEvent) -> str:
        x1, y1, x2, y2 = event.person.box
        cell = ((x1 + x2) // 96, (y1 + y2) // 96)
        return f"{event.zone.id}_{cell}"

    def update(self, events: list[ZoneBreachEvent], now: float) -> list[ZoneBreachEvent]:
        current_keys = set()
        confirmed: list[ZoneBreachEvent] = []
        for e in events:
            k = self._key(e)
            current_keys.add(k)
            self._streak[k] += 1
            if self._streak[k] >= self.required:
                last = self._last_alert_time.get(k, float("-inf"))
                if now - last >= self.cooldown_sec:
                    e.confirmed = True
                    confirmed.append(e)
                    self._last_alert_time[k] = now
        dead = [k for k in self._streak if k not in current_keys]
        for k in dead:
            del self._streak[k]
        return confirmed
