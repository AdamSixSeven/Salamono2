from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from backend.detector import Detection
from backend.marker_scheduler import MarkerScheduler
from backend.performance_profiler import PerformanceProfiler
from backend.routes import depth3d, ingest
from backend.routes.runtime import RuntimeOptionsPatch, patch_runtime_options
from backend.runtime_options import RuntimeProcessingOptions, RuntimeProcessingStore
from backend.zones_store import Zone


CAMERA_ID = "gate-camera"
ALL_OFF = {
    "boxes": False,
    "posture": False,
    "zones": False,
    "markers": False,
    "distances": False,
    "worker_id": False,
    "ppe": False,
}


def _frame() -> np.ndarray:
    return np.zeros((48, 64, 3), dtype=np.uint8)


def _person() -> Detection:
    return Detection(
        class_id=0,
        class_name="person",
        category="person",
        box=(8, 5, 42, 45),
        confidence=0.91,
    )


def _vehicle() -> Detection:
    return Detection(
        class_id=2,
        class_name="car",
        category="vehicle",
        box=(4, 10, 50, 40),
        confidence=0.82,
    )


class _Detector:
    def __init__(self, detections=()):
        self.detections = list(detections)
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        return list(self.detections)


class _MarkerDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        return []


class _ZoneStore:
    def __init__(self, zones=()):
        self.zones = list(zones)
        self.calls = 0

    def for_camera(self, _camera_id):
        self.calls += 1
        return list(self.zones)


class _CalibrationStore:
    def get(self, _camera_id):
        return None


class _EvidenceSpy:
    def __init__(self, *, due=False):
        self.due = due
        self.should_sample_calls = 0
        self.push_calls = 0
        self.trigger_calls = 0

    def should_sample(self, _camera_id, _timestamp):
        self.should_sample_calls += 1
        return self.due

    def push(self, _camera_id, _frame, _timestamp):
        self.push_calls += 1
        return 0.0

    def trigger(self, _camera_id, _alert_id, _timestamp):
        self.trigger_calls += 1
        return None


def _forbidden(name):
    return Mock(side_effect=AssertionError(f"unexpected runtime call: {name}"))


def _site_request(
    *,
    options: dict[str, bool],
    detector: _Detector | None = None,
    evidence=None,
    zone_store: _ZoneStore | None = None,
    zone_detector=None,
    zone_temporal_filter=None,
    marker_detector: _MarkerDetector | None = None,
    marker_scheduler=None,
    posture_manager=None,
    posture_worker=None,
    worker_identifier=None,
    worker_id_worker=None,
    unidentified_monitor=None,
):
    runtime_store = RuntimeProcessingStore()
    runtime_store.update(CAMERA_ID, **options)
    detector = detector or _Detector()
    zone_store = zone_store or _ZoneStore()
    marker_detector = marker_detector or _MarkerDetector()
    state = SimpleNamespace(
        detector=detector,
        danger_detector=SimpleNamespace(
            evaluate=_forbidden("danger.evaluate"),
            dynamic_zones=_forbidden("danger.dynamic_zones"),
        ),
        temporal_filter=SimpleNamespace(update=_forbidden("danger.update")),
        frame_store=SimpleNamespace(save=_forbidden("frame_store.save")),
        alert_store=SimpleNamespace(append=_forbidden("alert_store.append")),
        evidence_recorder=evidence,
        zone_store=zone_store,
        zone_detector=zone_detector or SimpleNamespace(
            evaluate=_forbidden("zone.evaluate")
        ),
        zone_temporal_filter=zone_temporal_filter or SimpleNamespace(
            update=_forbidden("zone.update")
        ),
        calibration_store=_CalibrationStore(),
        posture_manager=posture_manager,
        posture_worker=posture_worker,
        unidentified_worker_monitor=unidentified_monitor,
        worker_store=None,
        worker_profile_cache=None,
        worker_identifier=worker_identifier,
        worker_id_worker=worker_id_worker,
        runtime_processing_store=runtime_store,
        debug_inject_person=None,
        marker_detector=marker_detector,
        marker_scheduler=marker_scheduler,
        marker_zone_cache={},
        frame_counter=0,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _handle_site(request, *, trace=None):
    return ingest._handle_site(
        request,
        _frame(),
        time.time(),
        time.monotonic(),
        CAMERA_ID,
        "",
        trace,
    )


def test_pos_off_does_not_submit_or_read_posture_worker():
    posture_manager = SimpleNamespace(
        available=True,
        behavior_available=True,
        process=_forbidden("posture.process"),
    )
    posture_worker = SimpleNamespace(
        submit_latest=_forbidden("posture.submit_latest"),
        get_latest=_forbidden("posture.get_latest"),
    )
    request = _site_request(
        options={**ALL_OFF, "boxes": True},
        detector=_Detector([_person()]),
        posture_manager=posture_manager,
        posture_worker=posture_worker,
    )

    result = _handle_site(request)

    assert len(result.detections) == 1
    assert result.posture_assessments == []
    assert result.confirmed_posture_alerts == []
    assert result.posture_available is False
    posture_manager.process.assert_not_called()
    posture_worker.submit_latest.assert_not_called()
    posture_worker.get_latest.assert_not_called()


def test_id_off_does_not_schedule_scan_read_cache_or_update_monitor():
    identifier = SimpleNamespace(
        available=True,
        process=_forbidden("worker_id.process"),
    )
    worker = SimpleNamespace(
        submit_latest=_forbidden("worker_id.submit_latest"),
        current_identities=_forbidden("worker_id.current_identities"),
        scanned_track_ids=_forbidden("worker_id.scanned_track_ids"),
        stats=_forbidden("worker_id.stats"),
    )
    monitor = SimpleNamespace(update=_forbidden("unidentified.update"))
    request = _site_request(
        options={**ALL_OFF, "boxes": True},
        detector=_Detector([_person()]),
        worker_identifier=identifier,
        worker_id_worker=worker,
        unidentified_monitor=monitor,
    )

    result = _handle_site(request)

    assert len(result.detections) == 1
    assert result.worker_identifications == []
    assert result.unidentified_workers == []
    assert result.worker_identification_available is False
    identifier.process.assert_not_called()
    worker.submit_latest.assert_not_called()
    worker.current_identities.assert_not_called()
    worker.scanned_track_ids.assert_not_called()
    worker.stats.assert_not_called()
    monitor.update.assert_not_called()


def test_checkpoint_boxes_alone_still_require_detector():
    boxes_only = RuntimeProcessingOptions(
        boxes=True,
        posture=False,
        zones=False,
        markers=False,
        distances=False,
        worker_id=False,
        ppe=False,
    )
    all_off = RuntimeProcessingOptions(
        boxes=False,
        posture=False,
        zones=False,
        markers=False,
        distances=False,
        worker_id=False,
        ppe=False,
    )

    assert boxes_only.requires_detector("checkpoint") is True
    assert all_off.requires_detector("checkpoint") is False


def test_no_people_skips_per_person_posture_and_worker_id_work():
    posture_manager = SimpleNamespace(
        available=True,
        behavior_available=True,
        process=_forbidden("posture.process"),
    )
    posture_worker = SimpleNamespace(
        submit_latest=_forbidden("posture.submit_latest"),
        get_latest=_forbidden("posture.get_latest"),
    )
    identifier = SimpleNamespace(
        available=True,
        process=_forbidden("worker_id.process"),
    )
    worker = SimpleNamespace(
        submit_latest=_forbidden("worker_id.submit_latest"),
        current_identities=_forbidden("worker_id.current_identities"),
        scanned_track_ids=_forbidden("worker_id.scanned_track_ids"),
        stats=Mock(return_value=SimpleNamespace(thread_alive=True)),
    )
    monitor = SimpleNamespace(update=Mock(return_value=[]))
    detector = _Detector([_vehicle()])
    request = _site_request(
        options={**ALL_OFF, "posture": True, "worker_id": True},
        detector=detector,
        posture_manager=posture_manager,
        posture_worker=posture_worker,
        worker_identifier=identifier,
        worker_id_worker=worker,
        unidentified_monitor=monitor,
    )

    result = _handle_site(request)

    assert detector.calls == 1
    assert result.detector_ran is True
    assert result.posture_assessments == []
    assert result.worker_identifications == []
    posture_manager.process.assert_not_called()
    posture_worker.submit_latest.assert_not_called()
    posture_worker.get_latest.assert_not_called()
    identifier.process.assert_not_called()
    worker.submit_latest.assert_not_called()
    worker.current_identities.assert_not_called()
    worker.scanned_track_ids.assert_not_called()
    # The cheap availability probe and empty temporal update remain valid;
    # neither performs MediaPipe nor an ArUco crop scan.
    worker.stats.assert_called_once_with()
    monitor.update.assert_called_once()
    monitor_args = monitor.update.call_args.args
    assert monitor_args[:4] == (CAMERA_ID, "site", [], [])
    assert isinstance(monitor_args[4], float)


def test_markers_without_active_zone_or_calibration_lease_do_no_cv_work():
    yolo = _Detector()
    marker_detector = _MarkerDetector()
    scheduler = MarkerScheduler(marker_detector)
    zones = _ZoneStore([])
    request = _site_request(
        options={**ALL_OFF, "markers": True},
        detector=yolo,
        zone_store=zones,
        marker_detector=marker_detector,
        marker_scheduler=scheduler,
    )

    result = _handle_site(request)

    assert result.detector_ran is False
    assert result.markers == []
    assert yolo.calls == 0
    assert marker_detector.calls == 0
    assert zones.calls == 1


def test_active_marker_zone_scans_when_marker_visual_layer_is_off():
    yolo = _Detector()
    marker_detector = _MarkerDetector()
    scheduler = MarkerScheduler(marker_detector)
    zones = _ZoneStore([Zone(marker_ids=[10, 20, 30], active=True)])
    zone_detector = SimpleNamespace(evaluate=Mock(return_value=[]))
    zone_filter = SimpleNamespace(update=Mock(return_value=[]))
    request = _site_request(
        options={**ALL_OFF, "zones": True, "markers": False},
        detector=yolo,
        zone_store=zones,
        zone_detector=zone_detector,
        zone_temporal_filter=zone_filter,
        marker_detector=marker_detector,
        marker_scheduler=scheduler,
    )

    result = _handle_site(request)

    assert result.detector_ran is True
    assert yolo.calls == 1
    assert marker_detector.calls == 1
    assert result.markers == []
    zone_detector.evaluate.assert_called_once()
    zone_filter.update.assert_called_once()


def test_calibration_lease_scans_when_marker_visual_layer_is_off():
    yolo = _Detector()
    marker_detector = _MarkerDetector()
    scheduler = MarkerScheduler(marker_detector)
    scheduler.touch_calibration_session(CAMERA_ID)
    request = _site_request(
        options=ALL_OFF,
        detector=yolo,
        marker_detector=marker_detector,
        marker_scheduler=scheduler,
    )

    result = _handle_site(request)

    assert result.detector_ran is False
    assert yolo.calls == 0
    assert marker_detector.calls == 1
    assert result.markers == []


@pytest.mark.parametrize("sample_due", [False, True])
def test_backend_overlay_is_built_only_for_due_evidence_sample(
    monkeypatch,
    sample_due,
):
    evidence = _EvidenceSpy(due=sample_due)
    annotation = Mock(side_effect=lambda frame, *_args: frame)
    monkeypatch.setattr(ingest, "_annotate_site", annotation)
    trace = PerformanceProfiler().new_trace()
    request = _site_request(
        options={**ALL_OFF, "posture": True},
        detector=_Detector([_vehicle()]),
        evidence=evidence,
        posture_manager=SimpleNamespace(available=False),
    )

    result = _handle_site(request, trace=trace)

    assert result.posture_assessments == []
    assert evidence.should_sample_calls == 1
    assert evidence.push_calls == int(sample_due)
    assert annotation.call_count == int(sample_due)
    assert trace.values["overlay_ms"] >= 0.0


def test_due_evidence_keeps_detection_annotation_when_live_boxes_are_off(
    monkeypatch,
):
    evidence = _EvidenceSpy(due=True)
    annotation = Mock(side_effect=lambda frame, *_args: frame)
    monkeypatch.setattr(ingest, "_annotate_site", annotation)
    request = _site_request(
        options={**ALL_OFF, "posture": True, "boxes": False},
        detector=_Detector([_person()]),
        evidence=evidence,
        posture_manager=SimpleNamespace(available=False),
    )

    result = _handle_site(request)

    assert result.detections == []
    assert evidence.push_calls == 1
    annotation.assert_called_once()
    evidence_detections = annotation.call_args.args[1]
    assert len(evidence_detections) == 1
    assert evidence_detections[0].category == "person"


class _LatestFrameStore:
    def __init__(self, frame):
        now = time.time()
        self.item = SimpleNamespace(
            frame=frame,
            width=frame.shape[1],
            height=frame.shape[0],
            timestamp=now,
            received_at=now,
        )

    def get(self, _camera_id, max_age_seconds=None):
        return self.item


class _DepthEstimator:
    model_id = "fake-depth"

    def __init__(self):
        self.calls = 0

    def infer(self, frame):
        self.calls += 1
        return np.full(frame.shape[:2], 2.0, dtype=np.float32)

    def status(self):
        return {"device": "cpu"}


class _DepthResultStore:
    def __init__(self):
        self.result = None

    def put(self, result):
        self.result = result

    def get(self, _camera_id):
        return self.result


class _DepthProfileStore:
    def ensure_active(self, _camera_id):
        return "default"

    def get_profile(self, _camera_id, _profile_id):
        return {"profile_id": "default"}

    def get_calibration(self, _camera_id, _profile_id):
        return None

    def get_observations(self, _camera_id, _profile_id):
        return []

    def list_profiles(self, _camera_id):
        return []

    def active_profile_id(self, _camera_id):
        return "default"

    def get_depth_metric_correction(self, _camera_id, _profile_id):
        return None


@pytest.mark.asyncio
async def test_all_layers_off_skip_detector_overlay_evidence_and_depth_until_infer(
    monkeypatch,
):
    detector = _Detector()
    evidence = _EvidenceSpy(due=True)
    depth_estimator = _DepthEstimator()
    trace = PerformanceProfiler().new_trace()
    annotation = _forbidden("backend overlay")
    monkeypatch.setattr(ingest, "_annotate_site", annotation)
    request = _site_request(
        options=ALL_OFF,
        detector=detector,
        evidence=evidence,
    )
    state = request.app.state
    frame = _frame()
    state.latest_frame_store = _LatestFrameStore(frame)
    state.depth3d_calibration_store = _CalibrationStore()
    state.depth3d_result_store = _DepthResultStore()
    state.depth3d_profile_store = _DepthProfileStore()
    state.depth3d_estimator = depth_estimator
    state.analysis_lock = threading.Lock()

    live_result = _handle_site(request, trace=trace)

    assert live_result.detector_ran is False
    assert detector.calls == 0
    assert depth_estimator.calls == 0
    assert evidence.should_sample_calls == 0
    assert evidence.push_calls == 0
    assert annotation.call_count == 0
    assert trace.values["overlay_ms"] == 0.0

    status_payload = await depth3d.status(request, CAMERA_ID)
    assert status_payload["model"] == {"device": "cpu"}
    assert status_payload["latest_result"] is None
    assert depth_estimator.calls == 0

    depth_payload = await depth3d.infer(CAMERA_ID, request)

    assert depth_payload["model_id"] == "fake-depth"
    assert depth_payload["depth_jpeg_b64"]
    assert depth_estimator.calls == 1


@pytest.mark.asyncio
async def test_runtime_pos_patch_disables_background_worker_immediately():
    worker = SimpleNamespace(set_camera_enabled=Mock())
    state = SimpleNamespace(
        runtime_processing_store=RuntimeProcessingStore(),
        posture_worker=worker,
        worker_id_worker=None,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    payload = await patch_runtime_options(
        CAMERA_ID,
        RuntimeOptionsPatch(posture=False),
        request,
    )

    assert payload["options"]["posture"] is False
    worker.set_camera_enabled.assert_called_once_with(CAMERA_ID, False)
