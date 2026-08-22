from backend.calibration import Calibration
from backend.danger_rules import DangerDetector
from backend.detector import Detection
from config import DangerConfig


def det(category, box, cls_id=0):
    return Detection(
        class_id=cls_id,
        class_name="person" if category == "person" else "machine",
        category=category,
        box=tuple(box),
        confidence=0.95,
    )


def calibration_cm_per_px():
    # At 640×480: normalized u/v map to 6.4×4.8 m, i.e. 1 cm/px.
    return Calibration(
        camera_id="cam",
        marker_ids=[1, 2, 3, 4],
        width_m=6.4,
        height_m=4.8,
        homography=[[6.4, 0.0, 0.0], [0.0, 4.8, 0.0], [0.0, 0.0, 1.0]],
        source_frame_width=640,
        source_frame_height=480,
    )


def test_metric_warning_band_no_longer_creates_an_event():
    cfg = DangerConfig(warning_distance_m=3.0, danger_distance_m=1.5)
    detector = DangerDetector(cfg)
    person = det("person", [0, 0, 100, 100])       # foot x=50 => 0.5m
    machine = det("vehicle", [300, 0, 400, 100], 7)  # footprint starts at 3m
    events = detector.evaluate([person, machine], 1.0, calibration_cm_per_px())
    assert events == []


def test_metric_danger_zone_is_the_only_alarm_level():
    detector = DangerDetector(DangerConfig(warning_distance_m=3.0, danger_distance_m=1.5))
    person = det("person", [120, 0, 180, 100])
    machine = det("vehicle", [260, 0, 360, 100], 7)
    events = detector.evaluate([person, machine], 1.0, calibration_cm_per_px())
    assert len(events) == 1
    assert events[0].severity == "DANGER"
    assert events[0].rule_name == "person_vehicle_danger_zone"


def test_dynamic_machine_zones_are_projected_for_each_machine():
    detector = DangerDetector(DangerConfig(dynamic_zone_points=20))
    machine = det("vehicle", [300, 100, 500, 300], 7)
    zones = detector.dynamic_zones([machine], 640, 480, calibration_cm_per_px())
    assert len(zones) == 1
    assert zones[0].severity == "DANGER"
    assert all(zone.calibrated for zone in zones)
    assert all(len(zone.polygon_px) == 20 for zone in zones)
