from backend.detector import Detection
from backend.latest_detection_store import LatestDetectionStore


def _detection(x=1):
    return Detection(0, "person", "person", (x, 2, x + 3, 8), 0.9)


def test_returns_only_exact_frame_and_isolates_mutable_detections():
    store = LatestDetectionStore(max_cameras=2)
    source = _detection()
    store.put("cam", 10.0, 960, 720, [source])
    source.box = (99, 99, 100, 100)

    assert store.get_exact("cam", 10.1, 960, 720) is None
    assert store.get_exact("cam", 10.0, 1280, 720) is None
    snapshot = store.get_exact("cam", 10.0, 960, 720)
    assert snapshot is not None
    assert snapshot.detections[0].box == (1, 2, 4, 8)

    snapshot.detections[0].box = (7, 7, 8, 8)
    again = store.get_exact("cam", 10.0, 960, 720)
    assert again.detections[0].box == (1, 2, 4, 8)


def test_bounds_cameras_and_remove_discards_snapshot():
    store = LatestDetectionStore(max_cameras=2)
    store.put("a", 1, 10, 10, [_detection(1)])
    store.put("b", 1, 10, 10, [_detection(2)])
    store.put("c", 1, 10, 10, [_detection(3)])

    assert store.get_exact("a", 1, 10, 10) is None
    assert store.get_exact("b", 1, 10, 10) is not None
    store.remove("b")
    assert store.get_exact("b", 1, 10, 10) is None
