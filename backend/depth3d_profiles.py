"""Persistent calibration profiles and checkerboard captures for Depth3D."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
import io
import zipfile
from urllib.parse import quote
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.depth3d import Depth3DCalibration, IntrinsicCalibrationError
from backend.depth3d_checkerboard import CheckerboardObservation, CheckerboardSpec


def _safe(value: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return cleaned[:96] or fallback


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Depth3DProfileStore:
    """Stores profiles, raw calibration views and the active calibration on disk."""

    def __init__(self, root_dir: str, *, legacy_calibration_path: str | None = None, max_views: int = 40):
        self.root = Path(root_dir)
        self.max_views = max(8, int(max_views))
        self.index_path = self.root / "index.json"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()
        if legacy_calibration_path:
            self._migrate_legacy(Path(legacy_calibration_path))

    def _load_index(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw.setdefault("active_profiles", {})
                return raw
        except Exception:
            pass
        return {"active_profiles": {}}

    def _save_index(self) -> None:
        _atomic_json(self.index_path, self._index)

    def _camera_dir(self, camera_id: str) -> Path:
        return self.root / _safe(camera_id, "camera")

    def _profile_dir(self, camera_id: str, profile_id: str) -> Path:
        return self._camera_dir(camera_id) / _safe(profile_id, "profile")

    def _profile_json(self, camera_id: str, profile_id: str) -> Path:
        return self._profile_dir(camera_id, profile_id) / "profile.json"

    def _views_dir(self, camera_id: str, profile_id: str) -> Path:
        return self._profile_dir(camera_id, profile_id) / "views"

    def _read_profile(self, camera_id: str, profile_id: str) -> dict[str, Any] | None:
        path = self._profile_json(camera_id, profile_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _write_profile(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = time.time()
        _atomic_json(self._profile_json(payload["camera_id"], payload["profile_id"]), payload)

    def _new_profile_payload(self, camera_id: str, profile_id: str, name: str, spec: CheckerboardSpec | None) -> dict[str, Any]:
        now = time.time()
        return {
            "camera_id": camera_id,
            "profile_id": profile_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "checkerboard_spec": asdict(spec or CheckerboardSpec()),
            "calibration": None,
            "depth_metric_points": [],
            "depth_metric_correction": None,
        }

    def create_profile(self, camera_id: str, name: str, spec: CheckerboardSpec | None = None, *, profile_id: str | None = None, activate: bool = True) -> dict[str, Any]:
        spec = (spec or CheckerboardSpec()).validate()
        with self._lock:
            base = _safe(profile_id or name or "profil", "profil")
            candidate = base
            while self._profile_json(camera_id, candidate).exists():
                candidate = f"{base}-{uuid.uuid4().hex[:6]}"
            payload = self._new_profile_payload(camera_id, candidate, str(name or candidate).strip() or candidate, spec)
            self._write_profile(payload)
            self._views_dir(camera_id, candidate).mkdir(parents=True, exist_ok=True)
            if activate:
                self._index["active_profiles"][camera_id] = candidate
                self._save_index()
            return self.profile_summary(payload)

    def ensure_active(self, camera_id: str, spec: CheckerboardSpec | None = None) -> str:
        with self._lock:
            active = self._index.get("active_profiles", {}).get(camera_id)
            if active and self._read_profile(camera_id, active):
                return active
            profiles = self.list_profiles(camera_id)
            if profiles:
                active = profiles[0]["profile_id"]
                self._index["active_profiles"][camera_id] = active
                self._save_index()
                return active
            created = self.create_profile(camera_id, "Domyślny", spec, profile_id="default", activate=True)
            return created["profile_id"]

    def active_profile_id(self, camera_id: str) -> str | None:
        with self._lock:
            profile_id = self._index.get("active_profiles", {}).get(camera_id)
            return profile_id if profile_id and self._read_profile(camera_id, profile_id) else None

    def activate(self, camera_id: str, profile_id: str) -> dict[str, Any]:
        with self._lock:
            payload = self._read_profile(camera_id, profile_id)
            if not payload:
                raise KeyError(profile_id)
            self._index["active_profiles"][camera_id] = profile_id
            self._save_index()
            return self.profile_summary(payload)

    def rename(self, camera_id: str, profile_id: str, name: str) -> dict[str, Any]:
        with self._lock:
            payload = self._read_profile(camera_id, profile_id)
            if not payload:
                raise KeyError(profile_id)
            clean = str(name or "").strip()
            if not clean:
                raise ValueError("Nazwa profilu nie może być pusta.")
            payload["name"] = clean[:120]
            self._write_profile(payload)
            return self.profile_summary(payload)

    def delete_profile(self, camera_id: str, profile_id: str) -> None:
        with self._lock:
            path = self._profile_dir(camera_id, profile_id)
            if not path.exists():
                raise KeyError(profile_id)
            shutil.rmtree(path)
            active = self._index.get("active_profiles", {}).get(camera_id)
            if active == profile_id:
                self._index["active_profiles"].pop(camera_id, None)
                profiles = self.list_profiles(camera_id)
                if profiles:
                    self._index["active_profiles"][camera_id] = profiles[0]["profile_id"]
            self._save_index()

    def profile_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        camera_id = payload["camera_id"]
        profile_id = payload["profile_id"]
        views = self.list_views(camera_id, profile_id)
        calibration = payload.get("calibration")
        depth_metric_points = payload.get("depth_metric_points") if isinstance(payload.get("depth_metric_points"), list) else []
        depth_metric_correction = payload.get("depth_metric_correction") if isinstance(payload.get("depth_metric_correction"), dict) else None
        return {
            "camera_id": camera_id,
            "profile_id": profile_id,
            "name": payload.get("name", profile_id),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "checkerboard_spec": payload.get("checkerboard_spec", {}),
            "view_count": len(views),
            "calibrated": isinstance(calibration, dict),
            "calibration": calibration,
            "depth_metric_point_count": len(depth_metric_points),
            "depth_metric_correction": depth_metric_correction,
            "active": self.active_profile_id(camera_id) == profile_id,
        }

    def list_profiles(self, camera_id: str) -> list[dict[str, Any]]:
        with self._lock:
            directory = self._camera_dir(camera_id)
            if not directory.exists():
                return []
            records = []
            for item in directory.iterdir():
                if not item.is_dir():
                    continue
                payload = self._read_profile(camera_id, item.name)
                if payload:
                    records.append(self.profile_summary(payload))
            records.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
            return records

    def get_profile(self, camera_id: str, profile_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            profile_id = profile_id or self.active_profile_id(camera_id)
            if not profile_id:
                return None
            payload = self._read_profile(camera_id, profile_id)
            return self.profile_summary(payload) if payload else None

    def _view_paths(self, camera_id: str, profile_id: str, view_id: str) -> tuple[Path, Path]:
        base = self._views_dir(camera_id, profile_id) / _safe(view_id, "view")
        return base.with_suffix(".json"), base.with_suffix(".jpg")

    def add_view(self, camera_id: str, observation: CheckerboardObservation, frame_bgr: np.ndarray, spec: CheckerboardSpec, profile_id: str | None = None) -> tuple[bool, str | None, dict[str, Any]]:
        spec.validate()
        with self._lock:
            profile_id = profile_id or self.ensure_active(camera_id, spec)
            payload = self._read_profile(camera_id, profile_id)
            if not payload:
                raise KeyError(profile_id)
            saved_spec = CheckerboardSpec(**payload.get("checkerboard_spec", asdict(spec))).validate()
            if asdict(saved_spec) != asdict(spec):
                if self.list_views(camera_id, profile_id):
                    raise IntrinsicCalibrationError("Parametry szachownicy różnią się od aktywnego profilu. Utwórz nowy profil albo usuń zapisane widoki.")
                payload["checkerboard_spec"] = asdict(spec)

            signature = observation.pose_signature()
            for existing in self.get_observations(camera_id, profile_id):
                old = existing.pose_signature()
                center_distance = ((signature[0] - old[0]) ** 2 + (signature[1] - old[1]) ** 2) ** 0.5
                scale_distance = abs(signature[2] - old[2])
                angle_distance = abs(np.arctan2(np.sin(signature[3] - old[3]), np.cos(signature[3] - old[3])))
                if center_distance < 0.035 and scale_distance < 0.025 and angle_distance < np.deg2rad(6):
                    return False, "Widok jest zbyt podobny do już zapisanej klatki. Zmień pozycję lub kąt planszy.", {}

            view_id = f"view-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
            json_path, jpg_path = self._view_paths(camera_id, profile_id, view_id)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            view_payload = {
                "view_id": view_id,
                "camera_id": camera_id,
                "profile_id": profile_id,
                "timestamp": observation.timestamp,
                "frame_width": observation.frame_width,
                "frame_height": observation.frame_height,
                "image_corners": observation.image_corners.reshape(-1, 2).astype(float).tolist(),
                "coverage_ratio": observation.coverage_ratio,
                "blur_score": observation.blur_score,
                "corner_count": observation.corner_count,
                "reprojection_error_px": None,
                "created_at": time.time(),
            }
            ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise IntrinsicCalibrationError("Nie udało się zapisać obrazu widoku kalibracyjnego.")
            jpg_path.write_bytes(encoded.tobytes())
            _atomic_json(json_path, view_payload)
            payload["checkerboard_spec"] = asdict(spec)
            payload["calibration"] = None
            self._write_profile(payload)

            views = self.list_views(camera_id, profile_id)
            for old in views[self.max_views:]:
                self.delete_view(camera_id, profile_id, old["view_id"])
            return True, None, self.get_view(camera_id, profile_id, view_id) or view_payload

    def list_views(self, camera_id: str, profile_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            profile_id = profile_id or self.active_profile_id(camera_id)
            if not profile_id:
                return []
            directory = self._views_dir(camera_id, profile_id)
            if not directory.exists():
                return []
            items = []
            for path in directory.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["image_url"] = f"/api/depth3d/{quote(camera_id, safe='')}/profiles/{quote(profile_id, safe='')}/views/{quote(payload['view_id'], safe='')}/image"
                    items.append(payload)
                except Exception:
                    continue
            items.sort(key=lambda item: float(item.get("created_at") or item.get("timestamp") or 0), reverse=True)
            return items

    def get_view(self, camera_id: str, profile_id: str, view_id: str) -> dict[str, Any] | None:
        json_path, _ = self._view_paths(camera_id, profile_id, view_id)
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["image_url"] = f"/api/depth3d/{quote(camera_id, safe='')}/profiles/{quote(profile_id, safe='')}/views/{quote(view_id, safe='')}/image"
            return payload
        except Exception:
            return None

    def image_path(self, camera_id: str, profile_id: str, view_id: str) -> Path | None:
        _, path = self._view_paths(camera_id, profile_id, view_id)
        return path if path.exists() else None

    def get_observations(self, camera_id: str, profile_id: str | None = None) -> list[CheckerboardObservation]:
        result = []
        for item in reversed(self.list_views(camera_id, profile_id)):
            result.append(CheckerboardObservation(
                camera_id=camera_id,
                timestamp=float(item["timestamp"]),
                frame_width=int(item["frame_width"]),
                frame_height=int(item["frame_height"]),
                image_corners=np.asarray(item["image_corners"], dtype=np.float32).reshape(-1, 1, 2),
                coverage_ratio=float(item.get("coverage_ratio", 0.0)),
                blur_score=float(item.get("blur_score", 0.0)),
            ))
        return result

    def delete_view(self, camera_id: str, profile_id: str, view_id: str) -> None:
        with self._lock:
            json_path, jpg_path = self._view_paths(camera_id, profile_id, view_id)
            if not json_path.exists() and not jpg_path.exists():
                raise KeyError(view_id)
            json_path.unlink(missing_ok=True)
            jpg_path.unlink(missing_ok=True)
            payload = self._read_profile(camera_id, profile_id)
            if payload:
                payload["calibration"] = None
                self._write_profile(payload)

    def clear_views(self, camera_id: str, profile_id: str | None = None) -> None:
        with self._lock:
            profile_id = profile_id or self.active_profile_id(camera_id)
            if not profile_id:
                return
            directory = self._views_dir(camera_id, profile_id)
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
            payload = self._read_profile(camera_id, profile_id)
            if payload:
                payload["calibration"] = None
                self._write_profile(payload)

    def save_calibration(self, camera_id: str, profile_id: str, calibration: Depth3DCalibration) -> None:
        with self._lock:
            payload = self._read_profile(camera_id, profile_id)
            if not payload:
                raise KeyError(profile_id)
            calibration.updated_at = time.time()
            payload["calibration"] = asdict(calibration)
            payload["checkerboard_spec"] = {
                key: calibration.board_spec[key]
                for key in ("inner_corners_x", "inner_corners_y", "square_length_m", "lens_model")
                if key in calibration.board_spec
            }
            self._write_profile(payload)
            errors = calibration.board_spec.get("per_view_errors_px", [])
            for item, error in zip(reversed(self.list_views(camera_id, profile_id)), errors):
                json_path, _ = self._view_paths(camera_id, profile_id, item["view_id"])
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                raw["reprojection_error_px"] = round(float(error), 6)
                _atomic_json(json_path, raw)

    def get_calibration(self, camera_id: str, profile_id: str | None = None) -> Depth3DCalibration | None:
        with self._lock:
            profile_id = profile_id or self.active_profile_id(camera_id)
            if not profile_id:
                return None
            payload = self._read_profile(camera_id, profile_id)
            raw = payload.get("calibration") if payload else None
            return Depth3DCalibration(**raw) if isinstance(raw, dict) else None

    def all_calibrations(self) -> list[Depth3DCalibration]:
        result = []
        with self._lock:
            for camera_dir in self.root.iterdir():
                if not camera_dir.is_dir():
                    continue
                camera_id = None
                for profile_dir in camera_dir.iterdir():
                    payload = self._read_profile(profile_dir.parent.name, profile_dir.name)
                    if payload:
                        camera_id = payload.get("camera_id")
                        break
                if camera_id:
                    calibration = self.get_calibration(camera_id)
                    if calibration:
                        result.append(calibration)
        return result


    def list_depth_metric_points(self, camera_id: str, profile_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            profile_id = profile_id or self.active_profile_id(camera_id)
            if not profile_id:
                return []
            payload = self._read_profile(camera_id, profile_id)
            items = payload.get("depth_metric_points", []) if payload else []
            return [dict(item) for item in items if isinstance(item, dict)]

    def get_depth_metric_correction(self, camera_id: str, profile_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            profile_id = profile_id or self.active_profile_id(camera_id)
            if not profile_id:
                return None
            payload = self._read_profile(camera_id, profile_id)
            correction = payload.get("depth_metric_correction") if payload else None
            return dict(correction) if isinstance(correction, dict) else None

    def add_depth_metric_point(
        self,
        camera_id: str,
        profile_id: str,
        *,
        x: float,
        y: float,
        measured_distance_m: float,
        predicted_depth_m: float,
        label: str | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            payload = self._read_profile(camera_id, profile_id)
            if not payload:
                raise KeyError(profile_id)
            items = payload.get("depth_metric_points")
            if not isinstance(items, list):
                items = []
            point = {
                "point_id": f"pt-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}",
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "measured_distance_m": round(float(measured_distance_m), 6),
                "predicted_depth_m": round(float(predicted_depth_m), 6),
                "label": (str(label or "").strip()[:80] or None),
                "frame_width": int(frame_width or 0),
                "frame_height": int(frame_height or 0),
                "created_at": time.time(),
            }
            items.append(point)
            payload["depth_metric_points"] = items
            payload["depth_metric_correction"] = None
            self._write_profile(payload)
            return dict(point)

    def delete_depth_metric_point(self, camera_id: str, profile_id: str, point_id: str) -> None:
        with self._lock:
            payload = self._read_profile(camera_id, profile_id)
            if not payload:
                raise KeyError(profile_id)
            items = payload.get("depth_metric_points")
            if not isinstance(items, list):
                raise KeyError(point_id)
            new_items = [item for item in items if item.get("point_id") != point_id]
            if len(new_items) == len(items):
                raise KeyError(point_id)
            payload["depth_metric_points"] = new_items
            payload["depth_metric_correction"] = None
            self._write_profile(payload)

    def clear_depth_metric_points(self, camera_id: str, profile_id: str) -> None:
        with self._lock:
            payload = self._read_profile(camera_id, profile_id)
            if not payload:
                raise KeyError(profile_id)
            payload["depth_metric_points"] = []
            payload["depth_metric_correction"] = None
            self._write_profile(payload)

    def save_depth_metric_correction(self, camera_id: str, profile_id: str, correction: dict[str, Any]) -> None:
        with self._lock:
            payload = self._read_profile(camera_id, profile_id)
            if not payload:
                raise KeyError(profile_id)
            payload["depth_metric_correction"] = dict(correction)
            self._write_profile(payload)


    def export_profile_zip(self, camera_id: str, profile_id: str) -> bytes:
        with self._lock:
            directory = self._profile_dir(camera_id, profile_id)
            if not directory.exists():
                raise KeyError(profile_id)
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in directory.rglob("*"):
                    if path.is_file():
                        archive.write(path, arcname=str(Path(_safe(camera_id, "camera")) / _safe(profile_id, "profile") / path.relative_to(directory)))
            return output.getvalue()

    def _migrate_legacy(self, path: Path) -> None:
        if not path.exists() or any(self.root.glob("*/*/profile.json")):
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            records = raw if isinstance(raw, list) else raw.get("calibrations", [])
            for record in records:
                calibration = Depth3DCalibration(**record)
                profile = self.create_profile(calibration.camera_id, "Importowana kalibracja", CheckerboardSpec(), profile_id="legacy", activate=True)
                self.save_calibration(calibration.camera_id, profile["profile_id"], calibration)
        except Exception as exc:
            print(f"[depth3d] legacy calibration migration skipped: {exc}")


class CaptureStoreAdapter:
    def __init__(self, profiles: Depth3DProfileStore):
        self.profiles = profiles

    def add(self, observation: CheckerboardObservation, frame_bgr: np.ndarray | None = None, spec: CheckerboardSpec | None = None):
        if frame_bgr is None:
            raise IntrinsicCalibrationError("Brak obrazu źródłowego widoku kalibracyjnego.")
        return self.profiles.add_view(observation.camera_id, observation, frame_bgr, spec or CheckerboardSpec())[:2]

    def get(self, camera_id: str):
        return self.profiles.get_observations(camera_id)

    def clear(self, camera_id: str):
        self.profiles.clear_views(camera_id)


class CalibrationStoreAdapter:
    def __init__(self, profiles: Depth3DProfileStore):
        self.profiles = profiles

    def get(self, camera_id: str):
        return self.profiles.get_calibration(camera_id)

    def put(self, calibration: Depth3DCalibration):
        profile_id = self.profiles.ensure_active(calibration.camera_id)
        self.profiles.save_calibration(calibration.camera_id, profile_id, calibration)

    def all(self):
        return self.profiles.all_calibrations()
