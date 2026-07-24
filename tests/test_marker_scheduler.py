from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.calibration import CalibrationStore
from backend.latest_frame_store import LatestFrameStore
from backend.marker_detector import MarkerDetection
from backend.marker_scheduler import MarkerScheduler
from backend.routes import calibration
from backend.routes.ingest import _scheduled_markers
from backend.zones_store import Zone


class _FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        marker_id = self.calls
        return [
            MarkerDetection(
                marker_id=marker_id,
                corners=[
                    (1.0, 1.0),
                    (2.0, 1.0),
                    (2.0, 2.0),
                    (1.0, 2.0),
                ],
            )
        ]


def _frame():
    return np.zeros((24, 32, 3), dtype=np.uint8)


def test_no_zone_and_no_calibration_session_means_zero_detector_work():
    now = [0.0]
    detector = _FakeDetector()
    scheduler = MarkerScheduler(detector, clock=lambda: now[0])

    assert scheduler.process("cam", _frame()) == []
    now[0] = 100.0
    assert scheduler.process("cam", _frame()) == []
    assert detector.calls == 0


def test_normal_mode_is_limited_to_one_fps_and_returns_safe_cached_copy():
    now = [10.0]
    detector = _FakeDetector()
    scheduler = MarkerScheduler(detector, clock=lambda: now[0])

    first = scheduler.process("cam", _frame(), marker_zones_active=True)
    assert detector.calls == 1
    assert first[0].marker_id == 1
    first[0].corners[0] = (999.0, 999.0)

    now[0] = 10.5
    cached = scheduler.process("cam", _frame(), marker_zones_active=True)
    assert detector.calls == 1
    assert cached[0].corners[0] == (1.0, 1.0)

    now[0] = 11.01
    fresh = scheduler.process("cam", _frame(), marker_zones_active=True)
    assert detector.calls == 2
    assert fresh[0].marker_id == 2


def test_preview_lease_enables_five_fps_then_expires():
    now = [20.0]
    detector = _FakeDetector()
    scheduler = MarkerScheduler(
        detector,
        calibration_session_ttl=3.0,
        clock=lambda: now[0],
    )

    scheduler.touch_calibration_session("phone")
    assert scheduler.calibration_session_active("phone")
    scheduler.process("phone", _frame())
    assert detector.calls == 1

    now[0] = 20.1
    scheduler.process("phone", _frame())
    assert detector.calls == 1

    now[0] = 20.21
    scheduler.process("phone", _frame())
    assert detector.calls == 2

    now[0] = 23.01
    assert not scheduler.calibration_session_active("phone")
    assert scheduler.process("phone", _frame()) == []
    assert detector.calls == 2


def test_force_scans_exact_frame_without_waiting_for_calibration_interval():
    now = [30.0]
    detector = _FakeDetector()
    scheduler = MarkerScheduler(detector, clock=lambda: now[0])

    scheduler.process("cam", _frame(), calibration=True)
    now[0] = 30.01
    forced = scheduler.process("cam", _frame(), force=True)

    assert detector.calls == 2
    assert forced[0].marker_id == 2


def test_camera_and_detection_caches_are_bounded_lru():
    now = [40.0]
    detector = _FakeDetector()
    scheduler = MarkerScheduler(
        detector,
        max_cameras=2,
        max_markers_per_camera=1,
        clock=lambda: now[0],
    )

    scheduler.process("a", _frame(), marker_zones_active=True)
    scheduler.process("b", _frame(), marker_zones_active=True)
    assert scheduler.camera_count == 2

    # Reading A makes it the most recently used entry, so C evicts B.
    assert scheduler.get_cached("a")
    scheduler.process("c", _frame(), marker_zones_active=True)
    assert scheduler.camera_count == 2
    assert scheduler.get_cached("b") == []
    assert scheduler.get_cached("a")
    assert scheduler.get_cached("c")


def test_concurrent_calls_share_one_due_detection():
    detector = _FakeDetector()
    scheduler = MarkerScheduler(detector, clock=lambda: 50.0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: scheduler.process(
                    "shared",
                    _frame(),
                    marker_zones_active=True,
                ),
                range(20),
            )
        )

    assert detector.calls == 1
    assert all(result[0].marker_id == 1 for result in results)


def test_ingest_enables_scheduler_only_for_active_marker_zone():
    now = [60.0]
    detector = _FakeDetector()
    scheduler = MarkerScheduler(detector, clock=lambda: now[0])
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                marker_detector=detector,
                marker_scheduler=scheduler,
            )
        )
    )
    regular = Zone(polygon=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    inactive_marker = Zone(marker_ids=[10, 20, 30], active=False)
    active_marker = Zone(marker_ids=[10, 20, 30], active=True)

    assert _scheduled_markers(request, _frame(), "cam", [regular]) == []
    assert _scheduled_markers(
        request, _frame(), "cam", [inactive_marker]
    ) == []
    assert detector.calls == 0

    detected = _scheduled_markers(
        request, _frame(), "cam", [active_marker]
    )
    assert detector.calls == 1
    assert detected[0].marker_id == 1


@pytest.mark.asyncio
async def test_existing_preview_endpoint_renews_calibration_session(tmp_path):
    now = [70.0]
    detector = _FakeDetector()
    scheduler = MarkerScheduler(detector, clock=lambda: now[0])
    api = FastAPI()
    api.include_router(calibration.router, prefix="/api")
    api.state.marker_detector = detector
    api.state.marker_scheduler = scheduler
    api.state.latest_frame_store = LatestFrameStore()
    api.state.calibration_store = CalibrationStore(
        str(tmp_path / "calibrations.json")
    )

    async with AsyncClient(
        transport=ASGITransport(app=api),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/calibration/phone/preview")

    assert response.status_code == 200
    assert response.json()["frame_available"] is False
    assert scheduler.calibration_session_active("phone")

    # The normal ingest call now receives the 5 FPS calibration lease even
    # though this camera has no marker-defined zone.
    result = scheduler.process("phone", _frame())
    assert detector.calls == 1
    assert result[0].marker_id == 1
