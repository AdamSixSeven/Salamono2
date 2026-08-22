import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.danger_rules import (
    DangerDetector,
    TemporalFilter,
    bbox_iou,
    bbox_min_distance,
)
from backend.detector import Detection
from config import DangerConfig


def _make_det(category, box, cls_id=0, conf=0.9):
    names = {0: "person", 2: "car", 7: "truck"}
    return Detection(
        class_id=cls_id,
        class_name=names.get(cls_id, "unknown"),
        category=category,
        box=tuple(box),
        confidence=conf,
    )


class TestBboxIou:
    def test_no_overlap(self):
        assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_full_overlap(self):
        assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_partial_overlap(self):
        iou = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.1 < iou < 0.2  # 25/(100+100-25) = 0.143

    def test_containment(self):
        iou = bbox_iou((0, 0, 100, 100), (25, 25, 75, 75))
        assert 0.2 < iou < 0.3  # 2500/10000 = 0.25


class TestBboxMinDistance:
    def test_overlapping(self):
        assert bbox_min_distance((0, 0, 10, 10), (5, 5, 15, 15)) == 0.0

    def test_horizontal_gap(self):
        dist = bbox_min_distance((0, 0, 10, 10), (20, 0, 30, 10))
        assert dist == 10.0

    def test_vertical_gap(self):
        dist = bbox_min_distance((0, 0, 10, 10), (0, 20, 10, 30))
        assert dist == 10.0

    def test_diagonal_gap(self):
        dist = bbox_min_distance((0, 0, 10, 10), (13, 14, 23, 24))
        assert 4.9 < dist < 5.1  # sqrt(9+16) = 5.0

    def test_touching(self):
        assert bbox_min_distance((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


class TestDangerDetector:
    def test_overlap_triggers_danger(self):
        cfg = DangerConfig(proximity_px=50, overlap_iou=0.01)
        dd = DangerDetector(cfg)
        person = _make_det("person", [50, 50, 150, 200], cls_id=0)
        truck = _make_det("vehicle", [100, 100, 300, 300], cls_id=7)
        events = dd.evaluate([person, truck], 0.0)
        assert len(events) == 1
        assert events[0].severity == "DANGER"
        assert events[0].rule_name == "person_vehicle_overlap"

    def test_intermediate_proximity_does_not_trigger_warning(self):
        cfg = DangerConfig(proximity_px=50, overlap_iou=0.01)
        dd = DangerDetector(cfg)
        person = _make_det("person", [0, 0, 50, 100], cls_id=0)
        truck = _make_det("vehicle", [80, 0, 200, 100], cls_id=7)
        events = dd.evaluate([person, truck], 0.0)
        assert events == []

    def test_far_apart_no_danger(self):
        cfg = DangerConfig(proximity_px=50, overlap_iou=0.01)
        dd = DangerDetector(cfg)
        person = _make_det("person", [0, 0, 50, 100], cls_id=0)
        truck = _make_det("vehicle", [500, 0, 600, 100], cls_id=7)
        events = dd.evaluate([person, truck], 0.0)
        assert len(events) == 0

    def test_no_vehicles_no_danger(self):
        cfg = DangerConfig()
        dd = DangerDetector(cfg)
        person = _make_det("person", [0, 0, 50, 100], cls_id=0)
        events = dd.evaluate([person], 0.0)
        assert len(events) == 0

    def test_multiple_pairs(self):
        cfg = DangerConfig(proximity_px=50, overlap_iou=0.01)
        dd = DangerDetector(cfg)
        p1 = _make_det("person", [0, 0, 50, 100], cls_id=0)
        p2 = _make_det("person", [100, 100, 150, 200], cls_id=0)
        truck = _make_det("vehicle", [100, 100, 300, 300], cls_id=7)
        events = dd.evaluate([p1, p2, truck], 0.0)
        assert len(events) >= 1


class TestTemporalFilter:
    def _make_event(self, person_box, vehicle_box, severity="DANGER"):
        from backend.danger_rules import DangerEvent
        return DangerEvent(
            rule_name="person_vehicle_overlap",
            person=_make_det("person", person_box),
            hazard=_make_det("vehicle", vehicle_box, cls_id=7),
            distance_px=0.0,
            overlap_iou=0.5,
            severity=severity,
            frame_timestamp=0.0,
        )

    def test_not_confirmed_before_threshold(self):
        tf = TemporalFilter(required=3, cooldown_sec=10.0)
        evt = self._make_event([50, 50, 150, 200], [100, 100, 300, 300])
        assert len(tf.update([evt], 1.0)) == 0
        assert len(tf.update([evt], 2.0)) == 0

    def test_confirmed_at_threshold(self):
        tf = TemporalFilter(required=3, cooldown_sec=10.0)
        evt = self._make_event([50, 50, 150, 200], [100, 100, 300, 300])
        tf.update([evt], 1.0)
        tf.update([evt], 2.0)
        confirmed = tf.update([evt], 3.0)
        assert len(confirmed) == 1

    def test_streak_resets_on_gap(self):
        tf = TemporalFilter(required=3, cooldown_sec=10.0)
        evt = self._make_event([50, 50, 150, 200], [100, 100, 300, 300])
        tf.update([evt], 1.0)
        tf.update([evt], 2.0)
        tf.update([], 3.0)  # gap
        tf.update([evt], 4.0)
        confirmed = tf.update([evt], 5.0)
        assert len(confirmed) == 0

    def test_cooldown_prevents_re_alert(self):
        tf = TemporalFilter(required=2, cooldown_sec=10.0)
        evt = self._make_event([50, 50, 150, 200], [100, 100, 300, 300])
        tf.update([evt], 1.0)
        confirmed1 = tf.update([evt], 2.0)
        assert len(confirmed1) == 1
        confirmed2 = tf.update([evt], 3.0)
        assert len(confirmed2) == 0  # cooldown

    def test_cooldown_expires(self):
        tf = TemporalFilter(required=2, cooldown_sec=5.0)
        evt = self._make_event([50, 50, 150, 200], [100, 100, 300, 300])
        tf.update([evt], 1.0)
        tf.update([evt], 2.0)  # confirmed
        confirmed = tf.update([evt], 8.0)  # after cooldown
        assert len(confirmed) == 1

    def test_camera_scopes_do_not_clear_each_others_streaks(self):
        tf = TemporalFilter(required=2, cooldown_sec=5.0)
        evt = self._make_event([50, 50, 150, 200], [100, 100, 300, 300])

        assert tf.update([evt], 1.0, "live") == []
        assert tf.update([evt], 1.1, "demo") == []
        assert len(tf.update([evt], 2.0, "live")) == 1
        assert len(tf.update([evt], 2.1, "demo")) == 1

    def test_reset_camera_only_discards_requested_scope(self):
        tf = TemporalFilter(required=2, cooldown_sec=5.0)
        evt = self._make_event([50, 50, 150, 200], [100, 100, 300, 300])
        tf.update([evt], 1.0, "live")
        tf.update([evt], 1.0, "demo")

        tf.reset_camera("demo")

        assert len(tf.update([evt], 2.0, "live")) == 1
        assert tf.update([evt], 2.0, "demo") == []
