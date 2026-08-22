from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from backend.performance_profiler import COHORT_WITH_PERSON, PerformanceProfiler
from backend.routes.diagnostics import router


def _app(profiler=None):
    app = FastAPI()
    if profiler is not None:
        app.state.performance_profiler = profiler
    app.include_router(router)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/diagnostics/performance"),
        ("post", "/api/diagnostics/performance/reset"),
    ],
)
async def test_performance_endpoints_return_503_without_profiler(method, path):
    transport = ASGITransport(app=_app())

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await getattr(client, method)(path)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Performance profiler is not initialized"
    }


@pytest.mark.asyncio
async def test_performance_endpoint_includes_runtime_components():
    profiler = PerformanceProfiler()
    profiler.record_frame(
        {"total_frame_ms": 12.0}, has_person=True, source="live"
    )
    app = _app(profiler)
    app.state.detector = SimpleNamespace(
        status=lambda: {
            "backend": "pytorch",
            "model": "yolo11n.pt",
            "device": "cuda:0",
        }
    )
    app.state.frame_processor = SimpleNamespace(
        stats=SimpleNamespace(
            submitted=9,
            replaced=2,
            dropped=1,
            processed=6,
            failed=0,
        ),
        queue_depth=1,
    )
    app.state.demo_video_service = SimpleNamespace(
        stats=lambda: {"demo_active_jobs": 1}
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/diagnostics/performance")

    assert response.status_code == 200
    payload = response.json()
    assert payload["frame_counts"][COHORT_WITH_PERSON] == 1
    assert payload["detector"] == {
        "backend": "pytorch",
        "model": "yolo11n.pt",
        "device": "cuda:0",
    }
    assert payload["frame_processor"] == {
        "submitted": 9,
        "replaced": 2,
        "dropped": 1,
        "processed": 6,
        "failed": 0,
        "queue_depth": 1,
    }
    assert payload["demo_video"] == {"demo_active_jobs": 1}


@pytest.mark.asyncio
async def test_reset_endpoint_clears_samples_frames_sources_and_drops():
    profiler = PerformanceProfiler()
    profiler.record_frame(
        {"yolo_inference_ms": 8.0}, has_person=True, source="live"
    )
    profiler.record_drop("replaced", 3)
    transport = ASGITransport(app=_app(profiler))

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/diagnostics/performance/reset")
        after_reset = await client.get("/api/diagnostics/performance")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "reset"
    performance = payload["performance"]
    assert sum(performance["frame_counts"].values()) == 0
    assert performance["sources"] == {}
    assert performance["dropped_frames"]["total"] == 0
    assert performance["all"]["yolo_inference_ms"] == {
        "samples": 0,
        "mean": None,
        "median": None,
        "p95": None,
        "max": None,
    }

    assert after_reset.status_code == 200
    assert sum(after_reset.json()["frame_counts"].values()) == 0
