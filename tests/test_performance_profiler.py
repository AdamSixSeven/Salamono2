import pytest

import backend.performance_profiler as performance_module
from backend.performance_profiler import (
    COHORT_UNKNOWN,
    COHORT_WITHOUT_PERSON,
    COHORT_WITH_PERSON,
    FramePerformanceTrace,
    PerformanceProfiler,
)


def test_snapshot_calculates_mean_median_interpolated_p95_and_max():
    profiler = PerformanceProfiler(window_size=20)
    for value in (10.0, 20.0, 30.0, 40.0, 50.0):
        profiler.record_frame(
            {"yolo_inference_ms": value},
            has_person=True,
            source="live",
        )

    summary = profiler.snapshot()["cohorts"][COHORT_WITH_PERSON][
        "yolo_inference_ms"
    ]

    assert summary == {
        "samples": 5,
        "mean": 30.0,
        "median": 30.0,
        "p95": 48.0,
        "max": 50.0,
    }


def test_snapshot_keeps_person_no_person_and_unknown_cohorts_separate():
    profiler = PerformanceProfiler()
    profiler.record_frame(
        {"total_frame_ms": 10.0}, has_person=True, source="live"
    )
    profiler.record_frame(
        {"total_frame_ms": 30.0}, has_person=False, source="demo"
    )
    profiler.record_frame(
        {"total_frame_ms": 50.0}, has_person=None, source="worker"
    )

    snapshot = profiler.snapshot()

    assert snapshot["frame_counts"] == {
        COHORT_WITH_PERSON: 1,
        COHORT_WITHOUT_PERSON: 1,
        COHORT_UNKNOWN: 1,
    }
    assert snapshot["cohorts"][COHORT_WITH_PERSON]["total_frame_ms"][
        "mean"
    ] == 10.0
    assert snapshot["cohorts"][COHORT_WITHOUT_PERSON]["total_frame_ms"][
        "mean"
    ] == 30.0
    assert snapshot["cohorts"][COHORT_UNKNOWN]["total_frame_ms"][
        "mean"
    ] == 50.0
    assert snapshot["all"]["total_frame_ms"] == {
        "samples": 3,
        "mean": 30.0,
        "median": 30.0,
        "p95": 48.0,
        "max": 50.0,
    }


def test_metric_window_is_bounded_while_frame_counter_remains_cumulative():
    profiler = PerformanceProfiler(window_size=10)
    for value in range(15):
        profiler.record_frame(
            {"queue_age_ms": float(value)},
            has_person=False,
            source="live",
        )

    snapshot = profiler.snapshot()
    summary = snapshot["cohorts"][COHORT_WITHOUT_PERSON]["queue_age_ms"]

    assert snapshot["window_size"] == 10
    assert snapshot["frame_counts"][COHORT_WITHOUT_PERSON] == 15
    assert snapshot["sources"] == {"live": 15}
    assert summary == {
        "samples": 10,
        "mean": 9.5,
        "median": 9.5,
        "p95": 13.55,
        "max": 14.0,
    }


def test_dropped_frame_counters_are_grouped_and_ignore_nonpositive_counts():
    profiler = PerformanceProfiler()

    profiler.record_drop("replaced", 2)
    profiler.record_drop("rejected")
    profiler.record_drop("ignored-zero", 0)
    profiler.record_drop("ignored-negative", -3)

    dropped = profiler.snapshot()["dropped_frames"]

    assert dropped["total"] == 3
    assert dropped["by_reason"] == {"replaced": 2, "rejected": 1}


def test_trace_can_only_be_finalized_once(monkeypatch):
    profiler = PerformanceProfiler()
    trace = FramePerformanceTrace(
        profiler,
        source="demo",
        created_at=100.0,
        queued_at=100.0,
        has_person=True,
    )
    trace.set("yolo_inference_ms", 7.0)
    trace.dropped_frames = 2
    monkeypatch.setattr(performance_module.time, "perf_counter", lambda: 100.025)

    trace.finish()
    trace.set("yolo_inference_ms", 999.0)
    trace.add("overlay_ms", 999.0)
    trace.finish(has_person=False)

    snapshot = profiler.snapshot()

    assert trace.finalized is True
    assert snapshot["frame_counts"][COHORT_WITH_PERSON] == 1
    assert snapshot["frame_counts"][COHORT_WITHOUT_PERSON] == 0
    assert snapshot["sources"] == {"demo": 1}
    assert snapshot["cohorts"][COHORT_WITH_PERSON]["yolo_inference_ms"][
        "mean"
    ] == 7.0
    assert snapshot["cohorts"][COHORT_WITH_PERSON]["overlay_ms"][
        "samples"
    ] == 0
    assert snapshot["cohorts"][COHORT_WITH_PERSON]["dropped_frames"][
        "mean"
    ] == 2.0
    assert snapshot["cohorts"][COHORT_WITH_PERSON]["total_frame_ms"][
        "mean"
    ] == pytest.approx(25.0)
