from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "distance.html").read_text(encoding="utf-8")
JS = (ROOT / "frontend" / "distance.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "distance.css").read_text(encoding="utf-8")


def test_distance_page_keeps_live_mode_and_adds_clip_mode():
    assert 'id="distanceLiveTab"' in HTML
    assert 'id="distanceClipTab"' in HTML
    assert 'id="distanceLivePanel"' in HTML
    assert 'id="distanceClipPanel"' in HTML
    for live_control in (
        "distanceCamera",
        "distanceMax",
        "distanceRate",
        "distanceStart",
        "distanceOnce",
        "distanceStop",
        "distanceImage",
        "distancePairs",
    ):
        assert f'id="{live_control}"' in HTML


def test_clip_mode_uses_native_video_seek_and_one_frame_action():
    assert '<video id="clipVideo" controls' in HTML
    assert 'id="clipSeek" type="range"' in HTML
    assert 'id="clipPreviousFrame"' in HTML
    assert 'id="clipNextFrame"' in HTML
    assert 'id="clipAnalyze"' in HTML
    assert "Analizuj tę klatkę" in HTML
    assert ".mp4,.mov,.avi,.mkv" in HTML
    assert 'id="clipResultImage"' in HTML
    assert 'id="clipPairs"' in HTML
    assert 'id="clipDetections"' in HTML
    assert 'distance.css?v=2.0.0' in HTML
    assert 'distance.js?v=2.0.0' in HTML


def test_clip_frontend_uses_random_access_api_without_whole_video_analysis():
    assert 'form.append("video", file, file.name);' in JS
    assert 'apiJson("/api/distance/videos"' in JS
    assert "return {time_sec: time};" in JS
    assert "video.readyState" in JS
    assert 'preview_max_width: 1280' in JS
    assert '/analyze`' in JS
    assert 'data.actual_time_sec' in JS
    assert 'data.frame_index' in JS
    assert 'data.preview_jpeg_b64' in JS
    assert "renderClipPairs(data.pairs);" in JS
    assert "renderClipDetections(data.detections);" in JS
    assert "setInterval(analyzeClip" not in JS
    assert "requestAnimationFrame(analyzeClip" not in JS


def test_clip_async_actions_cannot_overwrite_a_newer_asset():
    assert "const restoreGeneration = generation;" in JS
    assert "restoreGeneration !== generation || uploading || clipId" in JS
    assert "const requestGeneration = ++generation;" in JS
    assert "if (clipAnalysisController) clipAnalysisController.abort();" in JS
    assert "setClipControlsDisabled(true);" in JS
    assert "if (!clipId || analyzing || uploading) return;" in JS
    assert "playerTime === null ? clipPositionSeconds() : playerTime" in JS


def test_clip_deletion_is_only_explicit_or_conscious_replacement():
    delete_helper = JS[JS.index("async function deleteClipAsset"):]
    assert 'method: "DELETE"' in delete_helper
    assert "void deleteClipAsset(oldClipId)" in JS
    assert "await deleteClipAsset(currentId);" in JS
    assert 'clip.remove.addEventListener("click"' in JS
    assert 'window.confirm("Usunąć ten klip z serwera?")' in JS
    assert 'clip.video.removeAttribute("src");' in JS
    assert "const previousSource = clip.video.currentSrc || clip.video.src;" in JS
    assert "beforeunload" not in JS
    assert "pagehide" not in JS


def test_distance_assets_have_valid_javascript_and_responsive_css():
    assert "@media (max-width: 900px)" in CSS
    assert ".dist-video-wrap video" in CSS
    node = shutil.which("node")
    if node is None:
        return
    result = subprocess.run(
        [node, "--check"],
        input=JS,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
