"""Marker-defined zones: polygon resolved from live ArUco detections."""
import pytest

from backend.marker_detector import MarkerDetection
from backend.routes.ingest import _resolve_marker_zones
from backend.zones_store import Zone


def _mk_marker(mid: int, cx: float, cy: float) -> MarkerDetection:
    # Corners collapsed to the same point — resolver only uses the centre.
    pt = (cx, cy)
    return MarkerDetection(marker_id=mid, corners=[pt, pt, pt, pt])


def test_regular_zone_pass_through():
    z = Zone(id="z1", name="Test", polygon=[[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]])
    out = _resolve_marker_zones([z], [], "cam", {}, now=1.0, frame_w=100, frame_h=100)
    assert len(out) == 1
    assert out[0].polygon == [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]


def test_marker_zone_all_markers_visible():
    z = Zone(id="mz1", name="Wykop", marker_ids=[10, 20, 30, 40])
    markers = [
        _mk_marker(10, 10, 10),   # TL
        _mk_marker(20, 90, 10),   # TR
        _mk_marker(30, 90, 90),   # BR
        _mk_marker(40, 10, 90),   # BL
    ]
    cache = {}
    out = _resolve_marker_zones([z], markers, "cam", cache,
                                now=1.0, frame_w=100, frame_h=100)
    assert len(out) == 1
    poly = out[0].polygon
    assert poly == [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    # Cache was populated for fallback next frame.
    assert ("cam", "mz1") in cache


def test_marker_zone_falls_back_to_cache_within_ttl():
    z = Zone(id="mz1", name="Wykop", marker_ids=[10, 20, 30, 40])
    cache = {("cam", "mz1"): ([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], 100.0)}
    # No markers visible now, but cache is 1 s old (< 2 s TTL) — reuse.
    out = _resolve_marker_zones([z], [], "cam", cache,
                                now=101.0, frame_w=100, frame_h=100)
    assert len(out) == 1
    assert out[0].polygon == [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def test_marker_zone_dropped_when_cache_stale():
    z = Zone(id="mz1", name="Wykop", marker_ids=[10, 20, 30, 40])
    cache = {("cam", "mz1"): ([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], 100.0)}
    # 5 s later, no markers visible, cache older than TTL — drop zone.
    out = _resolve_marker_zones([z], [], "cam", cache,
                                now=105.0, frame_w=100, frame_h=100)
    assert out == []


def test_marker_zone_dropped_when_no_cache_and_partial_visibility():
    z = Zone(id="mz1", name="Wykop", marker_ids=[10, 20, 30, 40])
    # Only 3 of 4 markers visible → not enough, no cache → drop.
    markers = [_mk_marker(10, 10, 10), _mk_marker(20, 90, 10), _mk_marker(30, 90, 90)]
    out = _resolve_marker_zones([z], markers, "cam", {},
                                now=1.0, frame_w=100, frame_h=100)
    assert out == []


def test_multiple_zones_independent_caches():
    z_a = Zone(id="a", name="A", marker_ids=[1, 2, 3])
    z_b = Zone(id="b", name="B", marker_ids=[4, 5, 6])
    # Only zone A's markers visible.
    markers = [_mk_marker(1, 10, 10), _mk_marker(2, 90, 10), _mk_marker(3, 50, 90)]
    cache = {}
    out = _resolve_marker_zones([z_a, z_b], markers, "cam", cache,
                                now=1.0, frame_w=100, frame_h=100)
    assert [z.id for z in out] == ["a"]
    assert ("cam", "a") in cache
    assert ("cam", "b") not in cache
