import numpy as np

from backend.latest_frame_store import LatestFrameStore


def _frame(value: int, *, width: int = 4, height: int = 3) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_set_and_get_copy_mutable_frames():
    store = LatestFrameStore()
    source = _frame(17)

    assert store.set(
        "cam_a",
        source,
        123.5,
        "checkpoint",
        received_at=130.0,
        image_bytes=bytearray(b"raw-image"),
        content_type="image/jpeg",
    )
    source[:] = 99

    first = store.get("cam_a")
    assert first is not None
    assert np.all(first.frame == 17)
    assert not np.shares_memory(first.frame, source)
    assert first.timestamp == 123.5
    assert first.captured_at == 123.5
    assert first.received_at == 130.0
    assert (first.width, first.height) == (4, 3)
    assert first.mode == "checkpoint"
    assert first.image_bytes == b"raw-image"
    assert first.content_type == "image/jpeg"

    first.frame[:] = 42
    second = store.get("cam_a")
    assert second is not None
    assert second is not first
    assert np.all(second.frame == 17)
    assert not np.shares_memory(first.frame, second.frame)


def test_get_applies_optional_receive_age_limit():
    store = LatestFrameStore()
    store.set("cam_a", _frame(1), 10.0, "site", received_at=100.0)

    assert store.get("cam_a") is not None
    assert store.get("cam_a", max_age_seconds=5.0, now=105.0) is not None
    assert store.get("cam_a", max_age_seconds=5.0, now=105.001) is None


def test_default_received_at_uses_server_clock(monkeypatch):
    monkeypatch.setattr("backend.latest_frame_store.time.time", lambda: 500.0)
    store = LatestFrameStore()

    store.set("cam_a", _frame(1), 10.0, "site")

    item = store.get("cam_a")
    assert item is not None
    assert item.timestamp == 10.0
    assert item.received_at == 500.0


def test_store_is_bounded_and_updates_eviction_order():
    store = LatestFrameStore(max_cameras=2)
    store.set("cam_a", _frame(1), 1.0, "site")
    store.set("cam_b", _frame(2), 2.0, "site")
    store.set("cam_a", _frame(3), 3.0, "checkpoint")
    store.set("cam_c", _frame(4), 4.0, "site")

    assert len(store) == 2
    assert store.get("cam_b") is None
    assert np.all(store.get("cam_a").frame == 3)
    assert np.all(store.get("cam_c").frame == 4)


def test_oversized_optional_preview_does_not_drop_bgr_snapshot():
    store = LatestFrameStore(max_frame_bytes=64)

    assert store.set(
        "cam_a",
        _frame(7),
        1.0,
        "site",
        image_bytes=b"x" * 65,
        content_type="image/jpeg",
    )

    item = store.get("cam_a")
    assert item is not None
    assert np.all(item.frame == 7)
    assert item.image_bytes is None
    assert item.content_type is None


def test_oversized_decoded_frame_is_rejected_before_copy():
    frame = _frame(7)
    store = LatestFrameStore(max_frame_bytes=frame.nbytes - 1)

    assert store.set("cam_a", frame, 1.0, "site") is False
    assert store.get("cam_a") is None


def test_legacy_put_preserves_raw_preview_contract():
    store = LatestFrameStore()

    assert store.put(
        "legacy",
        b"legacy-raw-bytes",
        content_type="image/jpeg",
        width=5,
        height=2,
        captured_at=50.0,
        received_at=51.0,
    )

    item = store.get("legacy")
    assert item is not None
    assert item.frame.shape == (2, 5, 3)
    assert item.image_bytes == b"legacy-raw-bytes"
    assert item.captured_at == 50.0
