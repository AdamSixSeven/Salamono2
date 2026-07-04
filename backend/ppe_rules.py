from dataclasses import dataclass

from backend.detector import Detection
from config import CONFIG, PPEConfig


@dataclass
class PPEEvent:
    person: Detection
    hardhat: Detection | None
    vest: Detection | None
    missing: list[str]         # subset of {"hardhat", "vest"}
    severity: str              # "DANGER" if anything missing, else "OK"
    frame_timestamp: float
    confirmed: bool = False


def _bbox_center(box: tuple) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _point_in_box(pt: tuple[float, float], box: tuple) -> bool:
    x, y = pt
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _upper_body_box(person_box: tuple) -> tuple:
    """Top third — where a hardhat should be if worn."""
    x1, y1, x2, y2 = person_box
    return (x1, y1, x2, y1 + (y2 - y1) // 3)


def _torso_box(person_box: tuple) -> tuple:
    """Middle half — where a vest should be if worn."""
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return (x1, y1 + h // 5, x2, y1 + (4 * h) // 5)


def _box_center_inside(child: Detection, parent_box: tuple) -> bool:
    return _point_in_box(_bbox_center(child.box), parent_box)


class PPEChecker:
    """Per-frame PPE compliance check with checkpoint-entry gating."""

    def __init__(self, config: PPEConfig | None = None):
        self.cfg = config or CONFIG.ppe
        self._last_check_ts: float = 0.0

    def evaluate(
        self,
        detections: list[Detection],
        frame_h: int,
        frame_timestamp: float,
    ) -> list[PPEEvent]:
        persons = [d for d in detections if d.category == "person"]
        hardhats = [d for d in detections if d.category == "hardhat"]
        vests = [d for d in detections if d.category == "vest"]

        events = []
        for p in persons:
            h_person = p.box[3] - p.box[1]
            if frame_h > 0 and h_person / frame_h < self.cfg.min_person_height_frac:
                continue

            head_zone = _upper_body_box(p.box)
            torso_zone = _torso_box(p.box)

            matched_hardhat = next(
                (h for h in hardhats if _box_center_inside(h, head_zone)),
                None,
            )
            matched_vest = next(
                (v for v in vests if _box_center_inside(v, torso_zone)),
                None,
            )

            missing = []
            if matched_hardhat is None:
                missing.append("hardhat")
            if matched_vest is None:
                missing.append("vest")

            events.append(PPEEvent(
                person=p,
                hardhat=matched_hardhat,
                vest=matched_vest,
                missing=missing,
                severity="DANGER" if missing else "OK",
                frame_timestamp=frame_timestamp,
            ))
        return events

    def confirm(self, events: list[PPEEvent], now: float) -> list[PPEEvent]:
        """Apply an entry cooldown so we don't alarm every frame for the same worker."""
        if not events:
            return []
        if now - self._last_check_ts < self.cfg.entry_cooldown_sec:
            return []
        self._last_check_ts = now
        confirmed = []
        for e in events:
            e.confirmed = True
            confirmed.append(e)
        return confirmed
