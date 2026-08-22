from types import SimpleNamespace

from backend.adaptive_yolo import AdaptiveYoloSizeController
from config import YOLOConfig


def _person(box=(10, 10, 50, 110)):
    return SimpleNamespace(category="person", box=box)


def test_uses_periodic_recovery_scan_after_consecutive_empty_frames():
    cfg = YOLOConfig(
        img_size=512,
        fall_recovery_img_size=640,
        fall_recovery_scan_interval_frames=3,
    )
    controller = AdaptiveYoloSizeController(cfg)

    for _ in range(3):
        assert controller.select_size("cam") == 512
        controller.observe("cam", [])

    assert controller.select_size("cam") == 640
    assert controller.status()["cameras"]["cam"]["last_reason"] == (
        "no_person_recovery_scan"
    )


def test_horizontal_person_arms_a_bounded_high_resolution_window():
    cfg = YOLOConfig(
        img_size=512,
        fall_recovery_img_size=640,
        fall_recovery_hold_frames=2,
        fall_recovery_horizontal_ratio=1.2,
    )
    controller = AdaptiveYoloSizeController(cfg)

    controller.observe("cam", [_person((0, 0, 180, 90))])
    assert controller.select_size("cam") == 640
    assert controller.select_size("cam") == 640
    assert controller.select_size("cam") == 512


def test_person_recovered_by_scan_keeps_recovery_size_for_following_frames():
    cfg = YOLOConfig(
        img_size=512,
        fall_recovery_img_size=640,
        fall_recovery_scan_interval_frames=1,
        fall_recovery_hold_frames=2,
    )
    controller = AdaptiveYoloSizeController(cfg)
    controller.observe("cam", [])
    assert controller.select_size("cam") == 640
    controller.observe("cam", [_person()])

    assert controller.select_size("cam") == 640
    assert controller.select_size("cam") == 640
    state = controller.status()["cameras"]["cam"]
    assert state["last_trigger"] == "person_recovered_at_high_resolution"
    assert state["recovery_frames"] == 3


def test_upright_person_stays_at_base_size():
    controller = AdaptiveYoloSizeController(YOLOConfig())
    controller.observe("cam", [_person()])
    assert controller.select_size("cam") == 512


def test_static_export_or_disabled_mode_never_changes_shape():
    cfg = YOLOConfig(
        adaptive_size_enabled=False,
        img_size=512,
        fall_recovery_img_size=640,
    )
    disabled = AdaptiveYoloSizeController(cfg)
    disabled.observe("cam", [_person((0, 0, 180, 90))])
    assert disabled.select_size("cam") == 512

    enabled = AdaptiveYoloSizeController(YOLOConfig())
    enabled.observe("cam", [_person((0, 0, 180, 90))])
    assert enabled.select_size(
        "cam",
        backend_supports_adaptive_size=False,
    ) == 512
