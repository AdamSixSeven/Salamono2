import base64
import os
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.danger_rules import DangerDetector, TemporalFilter
from backend.detector import Detector, PPE_CATEGORIES
from backend.frame_store import FrameStore
from backend.models import StatsOut
from backend.ppe_rules import PPEChecker
from backend.routes import alerts, ingest, ws
from backend.ws_manager import ConnectionManager
from config import CONFIG

PANEL_PASSWORD = os.getenv("PANEL_PASSWORD", "")
PUBLIC_PATHS = {"/api/health"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector()
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
            print(f"[startup] PPE checkpoint mode ready ({ppe_path})")
        except Exception as e:
            print(f"[startup] PPE model failed to load: {e}")
    else:
        print(f"[startup] PPE model not found at {ppe_path}; checkpoint mode disabled")
    app.state.ws_manager = ConnectionManager()
    app.state.frame_store = FrameStore()
    app.state.frame_counter = 0
    app.state.alert_history = []
    app.state.start_time = time.time()
    yield


app = FastAPI(title="Salamono Safety", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if not PANEL_PASSWORD or request.url.path in PUBLIC_PATHS:
        return await call_next(request)
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
        headers={"WWW-Authenticate": 'Basic realm="Salamono"'},
    )

app.include_router(ingest.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(ws.router)


@app.get("/api/stats", response_model=StatsOut)
async def get_stats():
    elapsed = time.time() - app.state.start_time
    fps = app.state.frame_counter / elapsed if elapsed > 0 else 0
    return StatsOut(
        total_frames_processed=app.state.frame_counter,
        total_alerts=len(app.state.alert_history),
        uptime_seconds=round(elapsed, 1),
        current_fps=round(fps, 2),
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/modes")
async def modes():
    return {
        "modes": ["site"] + (["checkpoint"] if app.state.ppe_detector else []),
        "checkpoint_available": app.state.ppe_detector is not None,
    }


os.makedirs(CONFIG.flagged_frames_dir, exist_ok=True)

app.mount("/static/flagged",
          StaticFiles(directory=CONFIG.flagged_frames_dir),
          name="flagged")

app.mount("/phone",
          StaticFiles(directory="phone", html=True),
          name="phone")

app.mount("/",
          StaticFiles(directory="frontend", html=True),
          name="frontend")
