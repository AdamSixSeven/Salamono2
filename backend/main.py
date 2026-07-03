import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.danger_rules import DangerDetector, TemporalFilter
from backend.detector import Detector
from backend.frame_store import FrameStore
from backend.models import StatsOut
from backend.routes import alerts, ingest, ws
from backend.ws_manager import ConnectionManager
from config import CONFIG


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector()
    app.state.danger_detector = DangerDetector()
    app.state.temporal_filter = TemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )
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
