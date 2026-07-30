import base64
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.alert_storage import AlertStore
from backend.calibration import CalibrationStore
from backend.camera_registry import CameraRegistry
from backend.danger_rules import DangerDetector, TemporalFilter
from backend.detector import Detector, PPE_CATEGORIES
from backend.depth3d import Depth3DResultStore, MonocularDepthEstimator
from backend.depth3d_profiles import (
    CalibrationStoreAdapter,
    CaptureStoreAdapter,
    Depth3DProfileStore,
)
from backend.worker_identification import WorkerIdentifier, UnidentifiedWorkerMonitor
from backend.worker_store import WorkerStore
from backend.frame_store import FrameStore
from backend.latest_frame_store import LatestFrameStore
from backend.evidence import EvidenceRecorder
from backend.frame_processor import LatestFrameProcessor
from backend.marker_detector import MarkerDetector
from backend.marker_scheduler import MarkerScheduler
from backend.models import StatsOut
from backend.ppe_rules import PPEChecker
from backend.posture_detector import PostureManager, PostureWorker
from backend.runtime_options import RuntimeProcessingStore
from backend.routes import alerts, calibration, depth3d, ingest, pair, reports, runtime, workers, ws, zones
from backend.ws_manager import ConnectionManager
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import ZoneStore
from config import CONFIG

logger = logging.getLogger(__name__)

ALERTS_LOG_PATH = os.path.join(CONFIG.flagged_frames_dir, "..", "alerts.jsonl")
ZONES_PATH = os.path.join(CONFIG.flagged_frames_dir, "..", "zones.json")
CALIBRATION_PATH = os.path.join(CONFIG.flagged_frames_dir, "..", "calibration.json")
CAMERA_ONLINE_MAX_AGE_SECONDS = 10.0

PANEL_PASSWORD = os.getenv("PANEL_PASSWORD", "")
DEMO_TOKEN = os.getenv("DEMO_TOKEN", "")
DEMO_COOKIE = "perimetr_demo"
DEMO_COOKIE_MAX_AGE = 12 * 60 * 60

PUBLIC_PATHS = {"/api/health"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector(categories={
        "person": list(CONFIG.yolo.person_class_ids),
        "vehicle": list(CONFIG.yolo.hazard_class_ids),
    })
    app.state.danger_detector = DangerDetector()
    app.state.temporal_filter = TemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )
    app.state.ppe_detector = None
    app.state.ppe_checker = None
    ppe_path = CONFIG.ppe.model_path
    if os.path.exists(ppe_path):
        try:
            app.state.ppe_detector = Detector(
                model_name=ppe_path,
                categories=PPE_CATEGORIES,
                confidence=CONFIG.ppe.confidence,
            )
            app.state.ppe_checker = PPEChecker()
            logger.info("PPE model loaded: %s", ppe_path)
        except Exception:
            logger.exception("PPE model failed to load: %s", ppe_path)
    else:
        logger.warning("PPE model not found: %s", ppe_path)
    app.state.ws_manager = ConnectionManager()
    app.state.frame_store = FrameStore()
    app.state.latest_frame_store = LatestFrameStore(max_cameras=16)
    app.state.runtime_processing_store = RuntimeProcessingStore(max_cameras=64)
    app.state.depth3d_profile_store = Depth3DProfileStore(
        CONFIG.depth3d.calibration_profiles_dir,
        legacy_calibration_path=CONFIG.depth3d.calibration_path,
        max_views=40,
    )
    app.state.depth3d_capture_store = CaptureStoreAdapter(app.state.depth3d_profile_store)
    app.state.depth3d_calibration_store = CalibrationStoreAdapter(app.state.depth3d_profile_store)
    app.state.depth3d_result_store = Depth3DResultStore(max_cameras=8)
    app.state.depth3d_estimator = MonocularDepthEstimator(
        enabled=CONFIG.depth3d.enabled,
        model_id=CONFIG.depth3d.model_id,
        device=CONFIG.depth3d.device,
        local_files_only=CONFIG.depth3d.local_files_only,
    )
    app.state.evidence_recorder = EvidenceRecorder()
    app.state.alert_store = AlertStore(ALERTS_LOG_PATH)
    app.state.zone_store = ZoneStore(ZONES_PATH)
    app.state.zone_detector = ZoneBreachDetector()
    app.state.zone_temporal_filter = ZoneTemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )
    app.state.marker_detector = MarkerDetector()
    app.state.marker_scheduler = MarkerScheduler(
        app.state.marker_detector,
        normal_fps=1.0,
        calibration_fps=5.0,
        calibration_session_ttl=3.0,
        max_cameras=16,
    )
    app.state.calibration_store = CalibrationStore(CALIBRATION_PATH)
    app.state.posture_manager = PostureManager()
    app.state.posture_worker = (
        PostureWorker(app.state.posture_manager)
        if app.state.posture_manager.available else None
    )
    app.state.posture_confirmations_seen = {}
    app.state.analysis_lock = threading.Lock()
    app.state.worker_identifier = WorkerIdentifier()
    app.state.unidentified_worker_monitor = UnidentifiedWorkerMonitor()
    app.state.worker_store = WorkerStore(CONFIG.worker_id.database_path)
    if app.state.posture_manager.available:
        logger.info(
            "Posture analysis ready: model=%s fps=%s",
            CONFIG.posture.model_path,
            f"{CONFIG.posture.sample_fps:g}",
        )
        if app.state.posture_manager.behavior_available:
            classifier = app.state.posture_manager.behavior_classifier
            logger.info(
                "Behavior classifier ready: model=%s device=%s",
                CONFIG.posture.behavior_model_path,
                classifier.device,
            )
        elif CONFIG.posture.behavior_enabled:
            logger.warning(
                "Behavior classifier unavailable: %s",
                getattr(app.state.posture_manager, "behavior_unavailable_reason", None),
            )
    else:
        logger.warning(
            "Posture analysis unavailable: %s",
            app.state.posture_manager.unavailable_reason,
        )
    if app.state.worker_identifier.available:
        logger.info(
            "Worker marker detection ready: fps=%s format=ArUco-4x4",
            f"{CONFIG.worker_id.sample_fps:g}",
        )
    else:
        logger.warning(
            "Worker marker detection unavailable: %s",
            app.state.worker_identifier.unavailable_reason,
        )
    app.state.marker_zone_cache = {}
    app.state.frame_counter = 0
    app.state.camera_registry = CameraRegistry(max_cameras=32)
    app.state.start_time = time.time()
    app.state.frame_processor = LatestFrameProcessor(
        process=lambda job: ingest.process_frame_job(app, job),
        publish=lambda job, result: ingest.publish_frame_result(
            app, job, result,
        ),
        on_error=lambda job, exc: ingest.record_frame_failure(app, job, exc),
        max_cameras=16,
        # The shared YOLO/PPE models and temporal filters form one inference
        # pipeline. Other cameras keep a replaceable newest-only slot while it
        # is busy instead of occupying blocked worker threads with stale work.
        max_parallel_cameras=1,
    )
    await app.state.frame_processor.start()
    try:
        yield
    finally:
        await app.state.frame_processor.close()
        if app.state.posture_worker is not None:
            # Never close MediaPipe native resources while its worker still
            # owns an active detect_for_video call.
            app.state.posture_worker.close(wait=True, timeout=None)
        app.state.posture_manager.close()
        app.state.evidence_recorder.close()


app = FastAPI(title="Perimetr", version="2.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def frame_headers(request: Request, call_next):
    """Allow the panel to be embedded in an external demonstration page.

    ``frame-ancestors *`` is used because ``X-Frame-Options`` has no portable
    allow-all value across modern browsers.
    """
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", "frame-ancestors *")
    # Revalidate coupled frontend assets on every navigation.  Otherwise an
    # embedded/mobile browser can combine fresh HTML with an older stylesheet
    # and render newly added controls as unstyled native inputs.
    path = request.url.path.lower()
    if path == "/" or path.endswith((".html", ".css", ".js")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if not PANEL_PASSWORD or request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    # 1) Demo token via query param — sets a cookie so subsequent asset
    #    requests (style.css, tokens/msbp.css, app.js…) pass through.
    if DEMO_TOKEN and request.query_params.get("demo") == DEMO_TOKEN:
        response = await call_next(request)
        response.set_cookie(
            key=DEMO_COOKIE,
            value=DEMO_TOKEN,
            max_age=DEMO_COOKIE_MAX_AGE,
            httponly=True,
            secure=True,
            samesite="none",     # required for cross-origin iframe embeds
        )
        return response

    if DEMO_TOKEN and request.cookies.get(DEMO_COOKIE) == DEMO_TOKEN:
        return await call_next(request)

    # 3) Classic HTTP basic auth.
    header = request.headers.get("Authorization", "")
    if header.startswith("Basic "):
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8", "ignore")
            _, _, pw = decoded.partition(":")
            if secrets.compare_digest(pw, PANEL_PASSWORD):
                return await call_next(request)
        except Exception:
            pass

    return Response(
        status_code=401,
        content="Unauthorized",
        headers={"WWW-Authenticate": 'Basic realm="Perimetr"'},
    )

app.include_router(ingest.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(zones.router, prefix="/api")
app.include_router(calibration.router, prefix="/api")
app.include_router(depth3d.router, prefix="/api")
app.include_router(pair.router, prefix="/api")
app.include_router(workers.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(runtime.router, prefix="/api")
app.include_router(ws.router)


@app.get("/api/stats", response_model=StatsOut)
async def get_stats():
    elapsed = time.time() - app.state.start_time
    fps = app.state.frame_counter / elapsed if elapsed > 0 else 0
    return StatsOut(
        total_frames_processed=app.state.frame_counter,
        total_alerts=app.state.alert_store.count(),
        uptime_seconds=round(elapsed, 1),
        current_fps=round(fps, 2),
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/readiness")
async def readiness():
    """Operational checklist for a pilot/demo installation.

    `ready_for_demo` means the end-to-end flow is usable.  Metric distance is
    reported separately because a camera can be demo-ready while still using
    the explicitly labelled pixel fallback.
    """
    posture_manager = getattr(app.state, "posture_manager", None)
    worker_identifier = getattr(app.state, "worker_identifier", None)
    evidence_recorder = getattr(app.state, "evidence_recorder", None)
    calibration_store = getattr(app.state, "calibration_store", None)
    zone_store = getattr(app.state, "zone_store", None)
    calibrations = calibration_store.all() if calibration_store is not None else []
    calibrations_by_camera = {
        calibration.camera_id: calibration for calibration in calibrations
    }
    zone_cameras = zone_store.all_cameras() if zone_store is not None else {}
    zones_total = sum(len(zones) for zones in zone_cameras.values())
    registry = getattr(app.state, "camera_registry", {})
    now = time.time()
    camera_ids = (
        set(registry)
        | set(zone_cameras)
        | set(calibrations_by_camera)
    )
    camera_readiness = []
    for camera_id in camera_ids:
        camera = registry.get(camera_id, {})
        last_seen = float(camera.get("last_seen", 0.0) or 0.0)
        online = (
            last_seen > 0.0
            and now - last_seen <= CAMERA_ONLINE_MAX_AGE_SECONDS
        )
        frame_width = int(camera.get("width", 0) or 0)
        frame_height = int(camera.get("height", 0) or 0)
        configured = [
            zone for zone in zone_cameras.get(camera_id, [])
            if zone.active and (
                len(zone.polygon) >= 3 or len(zone.marker_ids) >= 3
            )
        ]
        zones_configured = bool(configured)
        calibration_record = calibrations_by_camera.get(camera_id)
        calibration_exists = calibration_record is not None
        calibration_warning = None
        calibration_valid = False
        if calibration_record is not None:
            if frame_width > 0 and frame_height > 0:
                calibration_warning = calibration_record.compatibility_warning(
                    frame_width, frame_height,
                )
                calibration_valid = calibration_warning is None
            else:
                calibration_warning = (
                    "Brak wymiarów ostatniej klatki; nie można potwierdzić "
                    "zgodności kalibracji."
                )
        metric_distance_ready = (
            online
            and zones_configured
            and calibration_exists
            and calibration_valid
        )
        camera_readiness.append({
            "camera_id": camera_id,
            "online": online,
            "mode": camera.get("mode"),
            "last_seen": last_seen or None,
            "frame_width": frame_width or None,
            "frame_height": frame_height or None,
            "zones_configured": zones_configured,
            "zone_count": len(configured),
            "calibration_exists": calibration_exists,
            "calibration_valid": calibration_valid,
            "calibration_warning": calibration_warning,
            "metric_distance_ready": metric_distance_ready,
        })
    camera_readiness.sort(
        key=lambda item: float(item.get("last_seen") or 0.0),
        reverse=True,
    )
    metric_ready_camera_ids = [
        item["camera_id"]
        for item in camera_readiness
        if item["metric_distance_ready"]
    ]
    components = {
        "person_and_vehicle_detection": getattr(app.state, "detector", None) is not None,
        "ppe_checkpoint": getattr(app.state, "ppe_detector", None) is not None,
        "posture_analysis": bool(posture_manager and posture_manager.available),
        "learned_behavior_classifier": bool(
            posture_manager and getattr(posture_manager, "behavior_available", False)
        ),
        "worker_qr": bool(worker_identifier and worker_identifier.available),
        "evidence_clips": bool(evidence_recorder and evidence_recorder.enabled),
        "operator_review": getattr(app.state, "alert_store", None) is not None,
        "multi_camera_registry": hasattr(app.state, "camera_registry"),
        "browser_video_demo": True,
        "configured_zones": zones_total > 0,
        "metric_calibration": bool(calibrations),
    }
    warnings = []
    if not components["ppe_checkpoint"]:
        warnings.append("Brak modelu PPE — tryb bramki jest wyłączony.")
    if not components["posture_analysis"]:
        reason = posture_manager.unavailable_reason if posture_manager else "moduł niezainicjalizowany"
        warnings.append(f"Analiza postury jest niedostępna: {reason}.")
    elif CONFIG.posture.behavior_enabled and not components["learned_behavior_classifier"]:
        reason = (
            getattr(posture_manager, "behavior_unavailable_reason", None)
            if posture_manager else "moduł niezainicjalizowany"
        )
        warnings.append(f"Klasyfikator zachowań TCN jest niedostępny: {reason}.")
    if not components["metric_calibration"]:
        warnings.append("Brak kalibracji — odległości działają w oznaczonym trybie pikselowym.")
    elif not metric_ready_camera_ids:
        warnings.append(
            "Istnieje kalibracja, ale żadna aktywna kamera nie ma jednocześnie "
            "zgodnego formatu obrazu i skonfigurowanej strefy."
        )
    if not components["configured_zones"]:
        warnings.append("Nie skonfigurowano żadnej statycznej strefy bezpieczeństwa.")
    if CONFIG.yolo.hazard_class_ids == [2, 5, 7]:
        warnings.append("Domyślny COCO wykrywa samochód/autobus/ciężarówkę; koparki i walce wymagają własnego modelu.")

    core = (
        components["person_and_vehicle_detection"]
        and components["operator_review"]
        and components["evidence_clips"]
    )
    return {
        "status": "ready" if core else "degraded",
        "ready_for_demo": core,
        "ready_for_metric_demo": core and bool(metric_ready_camera_ids),
        "components": components,
        "zones_total": zones_total,
        "zone_cameras": sorted(zone_cameras.keys()),
        "calibrated_cameras": sorted(cal.camera_id for cal in calibrations),
        "metric_ready_cameras": metric_ready_camera_ids,
        # `cameras` is the concise public shape requested by the calibration
        # readiness contract; the descriptive alias helps existing clients.
        "cameras": camera_readiness,
        "camera_readiness": camera_readiness,
        "warnings": warnings,
    }


@app.get("/api/cameras")
async def cameras():
    """Return live cameras together with persisted calibration state.

    The calibration page uses the same registry as the main panel.  Cameras
    with a saved calibration remain selectable while offline, so every
    camera_id keeps its own calibration record and can be inspected or
    cleared without opening a local browser webcam.
    """
    now = time.time()
    registry = getattr(app.state, "camera_registry", {})
    calibration_store = getattr(app.state, "calibration_store", None)
    calibrations = (
        {item.camera_id: item for item in calibration_store.all()}
        if calibration_store is not None
        else {}
    )
    camera_ids = set(registry) | set(calibrations)
    rows = []
    for camera_id in camera_ids:
        item = registry.get(camera_id, {})
        row = dict(item)
        row.setdefault("camera_id", camera_id)
        last_seen = float(item.get("last_seen", 0.0) or 0.0)
        row["online"] = bool(
            last_seen > 0.0
            and now - last_seen <= CAMERA_ONLINE_MAX_AGE_SECONDS
        )
        calibration = calibrations.get(camera_id)
        row["calibrated"] = calibration is not None
        row["calibration_created_at"] = (
            calibration.created_at if calibration is not None else None
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            bool(row.get("online")),
            float(row.get("last_seen", 0.0) or 0.0),
            str(row.get("camera_id", "")),
        ),
        reverse=True,
    )
    return {"cameras": rows}


@app.get("/api/modes")
async def modes():
    posture_manager = getattr(app.state, "posture_manager", None)
    return {
        "modes": ["site"] + (["checkpoint"] if app.state.ppe_detector else []),
        "checkpoint_available": app.state.ppe_detector is not None,
        "posture_available": bool(posture_manager and posture_manager.available),
        "posture_unavailable_reason": (
            posture_manager.unavailable_reason if posture_manager else "not initialized"
        ),
        "behavior_classifier_available": bool(
            posture_manager and getattr(posture_manager, "behavior_available", False)
        ),
        "behavior_classifier_unavailable_reason": (
            getattr(posture_manager, "behavior_unavailable_reason", None)
            if posture_manager else "not initialized"
        ),
        "behavior_classifier_model": CONFIG.posture.behavior_model_path,
        "worker_identification_available": bool(
            getattr(app.state, "worker_identifier", None)
            and app.state.worker_identifier.available
        ),
        "worker_id_required_at_checkpoint": CONFIG.worker_id.require_at_checkpoint,
        "worker_id_required_on_site": CONFIG.worker_id.require_on_site,
        "evidence_clips_enabled": bool(app.state.evidence_recorder.enabled),
    }


os.makedirs(CONFIG.flagged_frames_dir, exist_ok=True)
os.makedirs(CONFIG.evidence.clips_dir, exist_ok=True)

app.mount("/static/flagged",
          StaticFiles(directory=CONFIG.flagged_frames_dir),
          name="flagged")

app.mount("/static/clips",
          StaticFiles(directory=CONFIG.evidence.clips_dir),
          name="clips")

app.mount("/phone",
          StaticFiles(directory="phone", html=True),
          name="phone")

app.mount("/",
          StaticFiles(directory="frontend", html=True),
          name="frontend")
