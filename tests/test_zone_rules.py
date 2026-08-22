import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.detector import Detection
from backend.zone_rules import (
    ZoneBreachDetector,
    ZoneBreachEvent,
    ZoneTemporalFilter,
    point_in_polygon,
)
from backend.zones_store import Zone


UNIT_SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
RIGHT_HALF = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
TRIANGLE = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]


def _person(box=(100, 100, 200, 400), conf=0.9):
    return Detection(
        class_id=0, class_name="person", category="person",
        box=box, confidence=conf,
    )


def _vehicle(box=(300, 300, 500, 500)):
    return Detection(
        class_id=7, class_name="truck", category="vehicle",
        box=box, confidence=0.9,
    )


class TestPointInPolygon:
    def test_inside(self):
        assert point_in_polygon(0.5, 0.5, UNIT_SQUARE) is True

    def test_outside(self):
        assert point_in_polygon(1.5, 0.5, UNIT_SQUARE) is False
        assert point_in_polygon(-0.1, 0.5, UNIT_SQUARE) is False

    def test_inside_right_half(self):
        assert point_in_polygon(0.75, 0.5, RIGHT_HALF) is True

    def test_outside_right_half(self):
        assert point_in_polygon(0.25, 0.5, RIGHT_HALF) is False

    def test_triangle(self):
        assert point_in_polygon(0.5, 0.4, TRIANGLE) is True
        assert point_in_polygon(0.5, 1.1, TRIANGLE) is False

    def test_degenerate_polygon(self):
        # fewer than 3 vertices → always outside
        assert point_in_polygon(0.5, 0.5, [[0, 0], [1, 1]]) is False
        assert point_in_polygon(0.5, 0.5, []) is False


class TestZoneBreachDetector:
    def _zone(self, name="Z", severity="DANGER", polygon=RIGHT_HALF, active=True):
        return Zone(name=name, severity=severity, polygon=polygon, active=active)

    def test_person_in_zone_triggers_breach(self):
        # feet at (500, 480) in 640x480 → norm (0.78, 1.0) → inside RIGHT_HALF
        person = _person(box=(450, 280, 550, 460))
        d = ZoneBreachDetector()
        events = d.evaluate([person], [self._zone()], 640, 480, 1.0)
        assert len(events) == 1
        assert events[0].severity == "DANGER"
        assert events[0].zone.name == "Z"

    def test_person_outside_zone_no_breach(self):
        person = _person(box=(50, 280, 150, 460))  # feet at (100, 480) → left half
        d = ZoneBreachDetector()
        events = d.evaluate([person], [self._zone()], 640, 480, 1.0)
        assert events == []

    def test_vehicles_ignored(self):
        vehicle = _vehicle(box=(450, 280, 550, 460))
        d = ZoneBreachDetector()
        events = d.evaluate([vehicle], [self._zone()], 640, 480, 1.0)
        assert events == []

    def test_inactive_zone_ignored(self):
        person = _person(box=(450, 280, 550, 460))
        z = self._zone(active=False)
        d = ZoneBreachDetector()
        assert d.evaluate([person], [z], 640, 480, 1.0) == []

    def test_polygon_too_short_ignored(self):
        person = _person(box=(450, 280, 550, 460))
        z = self._zone(polygon=[[0.5, 0.0], [1.0, 0.0]])
        d = ZoneBreachDetector()
        assert d.evaluate([person], [z], 640, 480, 1.0) == []

    def test_multiple_persons_one_in_one_out(self):
        p_in = _person(box=(450, 280, 550, 460))
        p_out = _person(box=(50, 280, 150, 460))
        d = ZoneBreachDetector()
        events = d.evaluate([p_in, p_out], [self._zone()], 640, 480, 1.0)
        assert len(events) == 1

    def test_multiple_zones_report_each(self):
        person = _person(box=(450, 280, 550, 460))
        z1 = self._zone(name="A", polygon=RIGHT_HALF)
        z2 = self._zone(name="B", polygon=UNIT_SQUARE)  # covers everything
        d = ZoneBreachDetector()
        events = d.evaluate([person], [z1, z2], 640, 480, 1.0)
        assert len(events) == 2
        assert {e.zone.name for e in events} == {"A", "B"}

    def test_warning_severity_propagates(self):
        person = _person(box=(450, 280, 550, 460))
        z = self._zone(severity="WARNING")
        d = ZoneBreachDetector()
        events = d.evaluate([person], [z], 640, 480, 1.0)
        assert events[0].severity == "WARNING"

    def test_zero_frame_dims_safe(self):
        person = _person()
        d = ZoneBreachDetector()
        assert d.evaluate([person], [self._zone()], 0, 480, 1.0) == []
        assert d.evaluate([person], [self._zone()], 640, 0, 1.0) == []

    def test_uses_feet_not_centroid(self):
        # tall person centered horizontally on boundary but feet at bottom
        # box centered at x=320 (norm 0.5, boundary), feet at (320, 480) — boundary
        # push slightly right to be clearly inside
        person = _person(box=(324, 100, 340, 460))
        d = ZoneBreachDetector()
        events = d.evaluate([person], [self._zone()], 640, 480, 1.0)
        assert len(events) == 1


class TestZoneTemporalFilter:
    def _event(self, zone_id="z1", severity="DANGER",
               person_box=(450, 280, 550, 460), ts=0.0):
        z = Zone(id=zone_id, name="Z", severity=severity, polygon=RIGHT_HALF)
        return ZoneBreachEvent(zone=z, person=_person(box=person_box),
                                severity=severity, frame_timestamp=ts)

    def test_not_confirmed_before_threshold(self):
        tf = ZoneTemporalFilter(required=3, cooldown_sec=1.0)
        assert tf.update([self._event(ts=1.0)], 1.0) == []
        assert tf.update([self._event(ts=2.0)], 2.0) == []

    def test_confirmed_at_threshold(self):
        tf = ZoneTemporalFilter(required=3, cooldown_sec=1.0)
        tf.update([self._event(ts=1.0)], 1.0)
        tf.update([self._event(ts=2.0)], 2.0)
        confirmed = tf.update([self._event(ts=3.0)], 3.0)
        assert len(confirmed) == 1
        assert confirmed[0].confirmed is True

    def test_streak_resets_on_absence(self):
        tf = ZoneTemporalFilter(required=3, cooldown_sec=1.0)
        tf.update([self._event(ts=1.0)], 1.0)
        tf.update([self._event(ts=2.0)], 2.0)
        tf.update([], 3.0)
        tf.update([self._event(ts=4.0)], 4.0)
        assert tf.update([self._event(ts=5.0)], 5.0) == []

    def test_cooldown_suppresses_repeat(self):
        tf = ZoneTemporalFilter(required=1, cooldown_sec=10.0)
        c1 = tf.update([self._event(ts=1.0)], 1.0)
        c2 = tf.update([self._event(ts=2.0)], 2.0)
        assert len(c1) == 1
        assert len(c2) == 0

    def test_cooldown_expires(self):
        tf = ZoneTemporalFilter(required=1, cooldown_sec=2.0)
        tf.update([self._event(ts=1.0)], 1.0)
        c = tf.update([self._event(ts=5.0)], 5.0)
        assert len(c) == 1

    def test_different_zones_independent_streaks(self):
        tf = ZoneTemporalFilter(required=2, cooldown_sec=1.0)
        e_a = self._event(zone_id="a", ts=1.0)
        e_b = self._event(zone_id="b", ts=1.0)
        tf.update([e_a, e_b], 1.0)
        confirmed = tf.update([
            self._event(zone_id="a", ts=2.0),
            self._event(zone_id="b", ts=2.0),
        ], 2.0)
        # both confirmed independently
        assert len(confirmed) == 2
        assert {c.zone.id for c in confirmed} == {"a", "b"}

    def test_camera_scopes_are_independent_and_resettable(self):
        tf = ZoneTemporalFilter(required=2, cooldown_sec=1.0)
        tf.update([self._event(ts=1.0)], 1.0, "live")
        tf.update([self._event(ts=1.0)], 1.0, "demo")

        tf.reset_camera("demo")

        assert len(tf.update([self._event(ts=2.0)], 2.0, "live")) == 1
        assert tf.update([self._event(ts=2.0)], 2.0, "demo") == []


def test_signed_distance_to_polygon_inside_and_outside():
    from backend.zone_rules import signed_distance_to_polygon
    poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert signed_distance_to_polygon((15.0, 5.0), poly) == pytest.approx(5.0)
    assert signed_distance_to_polygon((5.0, 5.0), poly) == pytest.approx(-5.0)


def test_static_zone_emits_warning_before_entry():
    # Zone begins at x=320. Person feet at x=300: 20 px before boundary.
    person = _person(box=(280, 200, 320, 450))
    zone = Zone(
        name="Szyb",
        severity="DANGER",
        polygon=RIGHT_HALF,
        warning_distance_px=25.0,
    )
    events = ZoneBreachDetector().evaluate([person], [zone], 640, 480, 1.0)
    assert len(events) == 1
    event = events[0]
    assert event.rule_name == "zone_approach"
    assert event.severity == "WARNING"
    assert event.inside is False
    assert event.distance_px == pytest.approx(20.0, abs=1e-6)


def test_static_zone_metric_distance_uses_calibration():
    from backend.calibration import Calibration
    # 100 px = 1 metre in both axes.
    cal = Calibration(
        camera_id="cam",
        marker_ids=[1, 2, 3, 4],
        width_m=6.4,
        height_m=4.8,
        homography=[[6.4, 0.0, 0.0], [0.0, 4.8, 0.0], [0.0, 0.0, 1.0]],
        source_frame_width=640,
        source_frame_height=480,
    )
    person = _person(box=(260, 200, 300, 450))  # foot x=280, 40 px = .4 m before zone
    zone = Zone(
        name="Wykop",
        severity="DANGER",
        polygon=RIGHT_HALF,
        warning_distance_m=0.5,
        warning_distance_px=1.0,
    )
    events = ZoneBreachDetector().evaluate([person], [zone], 640, 480, 1.0, cal)
    assert len(events) == 1
    assert events[0].rule_name == "zone_approach"
    assert events[0].calibrated is True
    assert events[0].distance_m == pytest.approx(0.4, abs=1e-6)
