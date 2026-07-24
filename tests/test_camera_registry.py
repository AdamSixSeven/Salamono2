from backend.camera_registry import CameraRegistry


def test_camera_registry_evicts_oldest_and_refreshes_reassigned_camera():
    registry = CameraRegistry(max_cameras=2)
    registry["a"] = {"frames": 1}
    registry["b"] = {"frames": 1}
    registry["a"] = {"frames": 2}
    registry["c"] = {"frames": 1}

    assert list(registry) == ["a", "c"]
    assert registry["a"]["frames"] == 2


def test_camera_registry_bulk_update_respects_bound():
    registry = CameraRegistry(
        {
            "a": {},
            "b": {},
            "c": {},
        },
        max_cameras=2,
    )

    assert list(registry) == ["b", "c"]

