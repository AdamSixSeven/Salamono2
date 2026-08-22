from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
ZONES_JS = (ROOT / "frontend" / "zones.js").read_text(encoding="utf-8")


def test_demo_uploads_the_selected_file_once_with_progress():
    assert "const xhr = new XMLHttpRequest();" in APP_JS
    assert 'xhr.open("POST", "/api/demo-videos", true);' in APP_JS
    assert 'form.append("video", file, file.name);' in APP_JS
    assert 'form.append("camera_id", DEMO_CAMERA_ID);' in APP_JS
    assert 'form.append("mode", State.mode);' in APP_JS
    assert 'form.append("playback_mode", "realtime");' in APP_JS
    assert "xhr.upload.onprogress" in APP_JS
    assert 'id="demoVideoProgress"' in INDEX_HTML


def test_browser_no_longer_decodes_or_posts_individual_demo_frames():
    forbidden = (
        'document.createElement("video")',
        "createObjectURL",
        "revokeObjectURL",
        "loadedmetadata",
        "requestVideoFrameCallback",
        "cancelVideoFrameCallback",
        ".toBlob(",
        'fetch("/api/frame"',
        'form.append("image"',
        "demo-frame.jpg",
    )
    for fragment in forbidden:
        assert fragment not in APP_JS


def test_demo_controls_cover_playback_and_change_is_available_during_failures():
    for element_id in (
        "demoVideoBtn",
        "demoVideoStopBtn",
        "demoVideoChangeBtn",
        "demoVideoStatusPanel",
        "demoVideoStatusText",
    ):
        assert f'id="{element_id}"' in INDEX_HTML

    assert 'ready: "Odtwórz"' in APP_JS
    assert 'playing: "Pauza"' in APP_JS
    assert 'paused: "Wznów"' in APP_JS
    assert 'finished: "Odtwórz ponownie"' in APP_JS
    assert 'void requestDemoAction("play")' in APP_JS
    assert 'void requestDemoAction("pause")' in APP_JS
    assert 'void requestDemoAction("resume")' in APP_JS
    assert 'void requestDemoAction("restart")' in APP_JS
    assert 'void requestDemoAction("stop")' in APP_JS
    assert 'showChange: state !== "idle" && state !== "deleted"' in APP_JS
    assert "demoVideoChangeBtn.disabled = false;" in APP_JS


def test_loading_has_a_bounded_ready_or_failed_exit_and_change_remains_usable():
    assert "const DEMO_PREPARE_TIMEOUT_MS" in APP_JS
    assert "startDemoPreparationWatchdog(generation);" in APP_JS
    assert "markDemoFailed(" in APP_JS
    assert 'updateDemoControls("failed", demoJob);' in APP_JS
    assert "normalizeDemoBackendState(snapshot.status)" in APP_JS
    assert "updateDemoControls(nextState, demoJob);" in APP_JS
    assert '"uploaded", "loading", "ready"' in APP_JS
    assert 'demoVideoChangeBtn.addEventListener("click", changeDemoVideo);' in APP_JS


def _source_between(start: str, end: str) -> str:
    start_index = APP_JS.index(start)
    end_index = APP_JS.index(end, start_index)
    return APP_JS[start_index:end_index]


def test_status_polling_and_all_control_endpoints_use_the_job_id():
    assert '"/api/demo-videos/" + encodeURIComponent(jobId)' in APP_JS
    assert '"/api/demo-videos/" + encodeURIComponent(jobId) + "/" + action' in APP_JS
    assert 'method: "DELETE"' in APP_JS
    assert 'method: "POST"' in APP_JS
    assert "scheduleDemoStatusPoll" in APP_JS
    assert "deleteDemoJobBestEffort(oldJobId);" in APP_JS


def test_page_navigation_disposes_only_frontend_clients_without_deleting_job():
    lifecycle = _source_between(
        "function disposeDemoPageClient()",
        "function resumeDemoPageClient()",
    )
    assert 'window.addEventListener("beforeunload"' not in APP_JS
    assert 'window.addEventListener("pagehide", disposeDemoPageClient);' in APP_JS
    assert 'window.addEventListener("pageshow", resumeDemoPageClient);' in APP_JS
    assert "clearDemoPollTimer();" in lifecycle
    assert "clearDemoPreparationWatchdog();" in lifecycle
    assert "disconnectWebSocketForPageLifecycle();" in lifecycle
    assert "cancelDemoRequests" not in lifecycle
    assert "demoUploadXhr.abort" not in lifecycle
    assert "deleteDemoJobBestEffort" not in lifecycle
    assert 'method: "DELETE"' not in lifecycle
    # Deletion happens only after a replacement file was actually selected.
    assert APP_JS.count("deleteDemoJobBestEffort(oldJobId);") == 1


def test_active_demo_job_is_persisted_and_restored_without_restart():
    restore = _source_between(
        "async function restorePersistedDemoJob()",
        "function deleteDemoJobBestEffort(jobId)",
    )
    assert 'const DEMO_SESSION_STORAGE_KEY = "perimetr:demo-video-session";' in APP_JS
    assert "localStorage.setItem(DEMO_SESSION_STORAGE_KEY" in APP_JS
    assert "localStorage.getItem(DEMO_SESSION_STORAGE_KEY)" in APP_JS
    assert '"/api/demo-videos/" + encodeURIComponent(demoJobId)' in restore
    assert "applyDemoSnapshot(snapshot, generation)" in restore
    assert "selectCamera(demoCameraId);" in restore
    assert "scheduleDemoStatusPoll(0, generation);" in restore
    initial_page_setup = APP_JS[APP_JS.rindex("filterAlertsUI();"):]
    assert "connectWebSocket();" in initial_page_setup
    assert "void restorePersistedDemoJob();" in initial_page_setup


def test_same_file_can_be_selected_again_and_picker_cancel_keeps_the_old_job():
    assert 'demoVideoInput.value = "";' in APP_JS
    change = _source_between("function changeDemoVideo()", "function startDemoUpload(file)")
    replacement = _source_between("function startDemoUpload(file)", "async function requestDemoAction")
    assert "openDemoFilePicker();" in change
    assert "resetDemoClient" not in change
    assert "deleteDemoJobBestEffort" not in change
    assert "deleteDemoJobBestEffort(oldJobId);" in replacement


def test_demo_frames_are_filtered_by_job_run_and_monotonic_frame_index():
    assert 'String(data.job_id || "") !== demoJobId' in APP_JS
    assert "incomingRunId !== demoRunId" in APP_JS
    assert "frameIndex <= demoLastFrameIndex" in APP_JS
    assert "demoLastFrameIndex = frameIndex;" in APP_JS
    assert "pendingBinaryFrames.length = 0;" in APP_JS
    assert "connectWebSocket();" in APP_JS
    assert 'data.type === "frame"' in APP_JS
    assert "if (expectsNewRun)" in APP_JS
    assert "activateDemoRun(null, false, true);" in APP_JS


def test_camera_and_demo_run_changes_clear_pending_spatial_results():
    assert "function clearPendingSpatialResult()" in ZONES_JS
    assert "window.clearTimeout(pendingSpatialFallbackTimer);" in ZONES_JS
    assert "pendingSpatialFrame = null;" in ZONES_JS
    assert 'document.addEventListener("perimetr-camera-changed"' in ZONES_JS
    assert 'document.addEventListener("perimetr-demo-run-changed"' in ZONES_JS
    assert ZONES_JS.count("clearLiveOverlayState();") >= 2
