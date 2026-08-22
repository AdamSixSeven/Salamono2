from types import SimpleNamespace
import threading

from backend.routes.ingest import reset_camera_temporal_state


class _Resettable:
    def __init__(self):
        self.calls = []

    def reset_camera(self, camera_id, **kwargs):
        self.calls.append((camera_id, kwargs))
        return True


class _Latest:
    def __init__(self):
        self.removed = []

    def remove(self, camera_id):
        self.removed.append(camera_id)


def test_restart_reset_clears_only_the_demo_camera_temporal_state():
    reset_names = (
        "posture_worker",
        "worker_id_worker",
        "temporal_filter",
        "zone_temporal_filter",
        "ppe_checker",
        "unidentified_worker_monitor",
        "marker_scheduler",
        "evidence_recorder",
    )
    components = {name: _Resettable() for name in reset_names}
    latest = _Latest()
    state = SimpleNamespace(
        **components,
        posture_manager=None,
        analysis_lock=threading.Lock(),
        posture_confirmations_seen={"demo": {1}, "live": {2}},
        marker_zone_cache={
            ("demo", "z1"): ([], 1.0),
            ("live", "z2"): ([], 1.0),
        },
        latest_frame_store=latest,
        camera_registry={"demo": {"frames": 4}, "live": {"frames": 7}},
    )
    app = SimpleNamespace(state=state)

    reset_camera_temporal_state(app, "demo", timeout=2.0)

    for name, component in components.items():
        assert component.calls
        assert component.calls[0][0] == "demo", name
    assert components["posture_worker"].calls == [
        ("demo", {"timeout": 2.0})
    ]
    assert state.posture_confirmations_seen == {"live": {2}}
    assert state.marker_zone_cache == {("live", "z2"): ([], 1.0)}
    assert latest.removed == ["demo"]
    assert state.camera_registry == {"live": {"frames": 7}}
