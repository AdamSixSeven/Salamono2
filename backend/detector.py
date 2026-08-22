from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
import os
from pathlib import Path
import time

import numpy as np
from config import CONFIG, YOLOConfig


logger = logging.getLogger(__name__)


SITE_CATEGORIES: dict[str, list[int]] = {
    "person": [0],
    # Compact Perimetr scene-v3 IDs. Keeping one ``vehicle`` category preserves
    # the original danger-zone, distance and alert pipeline for every machine.
    "vehicle": list(range(1, 10)),
}

PPE_CATEGORIES: dict[str, list[int]] = {
    "person": [5],
    "hardhat": [0],
    "no_hardhat": [2],
    "vest": [7],
    "no_vest": [4],
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    category: str
    box: tuple  # (x1, y1, x2, y2)
    confidence: float
    track_id: int | None = None


@dataclass(frozen=True)
class DetectorTiming:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    img_size: int = 0


class Detector:
    """One long-lived Ultralytics model with a low-overhead batch-1 path."""

    def __init__(
        self,
        model_name: str | None = None,
        categories: dict[str, list[int]] | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        device: str | None = None,
        img_size: int | None = None,
        config: YOLOConfig | None = None,
    ):
        cfg = config or CONFIG.yolo
        # Lazy imports keep lightweight rules/tests independent from the GPU
        # runtime. Production still fails clearly if Detector is instantiated
        # without the required inference packages.
        import torch
        from ultralytics import YOLO

        self._torch = torch
        self._yolo_factory = YOLO
        self.cfg = cfg
        self.categories = categories or SITE_CATEGORIES
        self._class_to_category: dict[int, str] = {}
        for category, ids in self.categories.items():
            for class_id in ids:
                self._class_to_category[int(class_id)] = category
        self._class_filter = list(self._class_to_category)
        self.conf = (
            confidence
            if confidence is not None
            else cfg.confidence_threshold
        )
        self.iou = iou if iou is not None else cfg.iou_threshold
        self.device = self._resolve_device(device or cfg.device)
        self.img_size = int(img_size or cfg.img_size)
        self._backend_notes: list[str] = []
        self.requested_precision = str(cfg.precision or "auto").strip().lower()
        self.precision = self._resolve_precision(self.requested_precision)
        self.quantize = 16 if self.precision == "fp16" else 32
        self.requested_backend = str(cfg.backend or "auto").strip().lower()
        self.allow_backend_fallback = bool(cfg.allow_backend_fallback)
        self._fallback_used = False
        self._pytorch_model_path = str(model_name or cfg.model_name)
        self.model_path, self.selected_backend = self._select_model_path(
            explicit_model=model_name,
        )
        self.model = self._load_with_constructor_fallback(self.model_path)
        self._predictor = None
        self._warmed_up = False
        self._warmup_ms = 0.0
        self._fast_path_requested = bool(cfg.fast_predictor_enabled)
        self._fast_path_active = False
        self._fast_path_error: str | None = None
        self._actual_backend = self.selected_backend
        self._actual_device = self.device
        self._actual_precision = self.precision
        self._supports_adaptive_size = self.selected_backend == "pytorch"

        if str(self.device).startswith("cuda") or str(self.device).isdigit():
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    def _resolve_device(self, requested: str) -> str:
        value = str(requested or "auto").strip().lower()
        if value not in {"", "auto"}:
            return str(requested)
        return "0" if self._torch.cuda.is_available() else "cpu"

    def _resolve_precision(self, requested: str) -> str:
        value = str(requested or "auto").strip().lower()
        aliases = {
            "16": "fp16",
            "half": "fp16",
            "float16": "fp16",
            "32": "fp32",
            "float": "fp32",
            "float32": "fp32",
        }
        value = aliases.get(value, value)
        if value in {"fp16", "fp32"}:
            if value == "fp16" and not self._is_cuda_device():
                self._backend_notes.append(
                    "FP16 requested without CUDA; using FP32"
                )
                return "fp32"
            return value
        if value != "auto":
            raise ValueError("YOLO_PRECISION must be auto, fp32 or fp16")
        if not self._is_cuda_device():
            return "fp32"
        # Ampere/Ada generally benefit from FP16 at batch 1. On the deployed
        # Turing RTX 2070 SUPER profiling showed FP32 to be 5-7% faster.
        try:
            index = 0
            if ":" in str(self.device):
                index = int(str(self.device).split(":", 1)[1])
            elif str(self.device).isdigit():
                index = int(self.device)
            major, _minor = self._torch.cuda.get_device_capability(index)
            return "fp16" if major >= 8 else "fp32"
        except Exception:
            return "fp32"

    def _is_cuda_device(self) -> bool:
        value = str(self.device).strip().lower()
        # ``auto`` is resolved using torch.cuda.is_available() before this
        # method is called.  An explicit ``0``/``cuda:0`` is therefore treated
        # as an intentional CUDA request even in lightweight tests where no
        # physical GPU is exposed.
        return bool(value.isdigit() or value.startswith("cuda"))

    @staticmethod
    def _exists(path: str) -> bool:
        return bool(path and Path(path).is_file())

    @staticmethod
    def _tensorrt_available() -> bool:
        return importlib.util.find_spec("tensorrt") is not None

    @staticmethod
    def _onnx_cuda_available() -> bool:
        if importlib.util.find_spec("onnxruntime") is None:
            return False
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    def _optional_candidate(
        self,
        backend: str,
        path: str,
    ) -> tuple[str, str] | None:
        if not self._exists(path):
            self._backend_notes.append(f"{backend}: model file unavailable")
            return None
        if backend == "tensorrt" and not self._tensorrt_available():
            self._backend_notes.append("tensorrt: runtime unavailable")
            return None
        if backend == "onnx" and not self._onnx_cuda_available():
            self._backend_notes.append(
                "onnx: CUDAExecutionProvider unavailable"
            )
            return None
        return str(path), backend

    def _select_model_path(
        self,
        *,
        explicit_model: str | None,
    ) -> tuple[str, str]:
        if explicit_model:
            suffix = Path(explicit_model).suffix.lower()
            return str(explicit_model), {
                ".engine": "tensorrt",
                ".onnx": "onnx",
            }.get(suffix, "pytorch")

        backend = self.requested_backend
        if backend not in {"auto", "pytorch", "onnx", "tensorrt"}:
            raise ValueError(
                "YOLO_BACKEND must be auto, pytorch, onnx or tensorrt"
            )
        if backend == "tensorrt":
            candidate = self._optional_candidate(
                "tensorrt",
                self.cfg.tensorrt_model_path,
            )
            if candidate is not None:
                return candidate
        elif backend == "onnx":
            candidate = self._optional_candidate(
                "onnx",
                self.cfg.onnx_model_path,
            )
            if candidate is not None:
                return candidate
        elif backend == "auto":
            for name, path in (
                ("tensorrt", self.cfg.tensorrt_model_path),
                ("onnx", self.cfg.onnx_model_path),
            ):
                if not path:
                    continue
                candidate = self._optional_candidate(name, path)
                if candidate is not None:
                    return candidate

        if backend in {"onnx", "tensorrt"}:
            if not self.allow_backend_fallback:
                raise RuntimeError(
                    f"Requested YOLO backend {backend!r} is unavailable"
                )
            self._fallback_used = True
            self._backend_notes.append(
                f"{backend}: falling back to PyTorch"
            )
        return self._pytorch_model_path, "pytorch"

    def _load_with_constructor_fallback(self, model_path: str):
        try:
            return self._yolo_factory(model_path)
        except Exception as exc:
            if (
                self.selected_backend == "pytorch"
                or not self.allow_backend_fallback
            ):
                raise
            self._fallback_used = True
            self._backend_notes.append(
                f"{self.selected_backend}: load failed ({type(exc).__name__}); "
                "falling back to PyTorch"
            )
            self.model_path = self._pytorch_model_path
            self.selected_backend = "pytorch"
            return self._yolo_factory(self.model_path)

    def _predict_kwargs(self, img_size: int) -> dict:
        return {
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "imgsz": int(img_size),
            "classes": self._class_filter,
            # Current Ultralytics uses quantize; passing deprecated ``half``
            # produced one warning per frame in the previous implementation.
            "quantize": self.quantize,
            "verbose": False,
        }

    def warmup(self) -> None:
        """Build the predictor and complete all lazy CUDA initialization."""
        if self._warmed_up:
            return
        started = time.perf_counter()
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        attempts = max(1, int(self.cfg.warmup_runs))
        try:
            with self._torch.inference_mode():
                for _ in range(attempts):
                    self.model.predict(
                        dummy,
                        **self._predict_kwargs(self.img_size),
                    )
        except Exception as exc:
            if (
                self.selected_backend == "pytorch"
                or not self.allow_backend_fallback
            ):
                raise
            logger.warning(
                "YOLO %s warmup failed; falling back to PyTorch: %s",
                self.selected_backend,
                exc,
            )
            self._fallback_used = True
            self._backend_notes.append(
                f"{self.selected_backend}: warmup failed "
                f"({type(exc).__name__}); falling back to PyTorch"
            )
            self.model_path = self._pytorch_model_path
            self.selected_backend = "pytorch"
            self.model = self._yolo_factory(self.model_path)
            with self._torch.inference_mode():
                for _ in range(attempts):
                    self.model.predict(
                        dummy,
                        **self._predict_kwargs(self.img_size),
                    )

        self._predictor = getattr(self.model, "predictor", None)
        self._warmed_up = True
        self._configure_actual_backend()
        self._fast_path_active = bool(
            self._fast_path_requested
            and self._predictor is not None
            and callable(getattr(self._predictor, "preprocess", None))
            and callable(getattr(self._predictor, "inference", None))
            and callable(getattr(self._predictor, "postprocess", None))
        )
        if self._fast_path_active:
            # Ultralytics' PyTorch letterbox uses rectangular tensors. Warm the
            # three shapes used by phone landscape, widescreen demo and phone
            # portrait so the first user frame never pays lazy CUDA kernels.
            try:
                warm_sizes = {self.img_size}
                if bool(getattr(self.cfg, "adaptive_size_enabled", False)):
                    warm_sizes.add(int(self.cfg.fall_recovery_img_size))
                for warm_size in sorted(warm_sizes):
                    for height, width in (
                        (720, 960),
                        (540, 960),
                        (380, 672),
                        (960, 720),
                    ):
                        warm_frame = np.zeros((height, width, 3), dtype=np.uint8)
                        self._fast_predict(warm_frame, warm_size)
            except Exception as exc:
                self._fast_path_active = False
                self._fast_path_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Optimized YOLO warmup failed; public API remains active: %s",
                    exc,
                )
        self._warmup_ms = (time.perf_counter() - started) * 1000.0

    def _configure_actual_backend(self) -> None:
        predictor = self._predictor
        backend = getattr(predictor, "model", None)
        model_format = str(getattr(backend, "format", "") or "").lower()
        if model_format in {"engine", "tensorrt"}:
            self._actual_backend = "tensorrt"
        elif model_format == "onnx":
            self._actual_backend = "onnx"
        elif model_format:
            self._actual_backend = model_format
        else:
            self._actual_backend = self.selected_backend
        actual_device = getattr(backend, "device", None)
        if actual_device is not None:
            self._actual_device = str(actual_device)
        self._actual_precision = (
            "fp16" if bool(getattr(backend, "fp16", False)) else "fp32"
        )
        self._supports_adaptive_size = bool(
            self._actual_backend in {"pt", "pytorch"}
            or bool(getattr(backend, "dynamic", False))
        )

    def _set_predictor_size(self, img_size: int) -> int:
        predictor = self._predictor
        assert predictor is not None
        effective = int(img_size)
        backend = predictor.model
        if not self._supports_adaptive_size and hasattr(backend, "imgsz"):
            exported = getattr(backend, "imgsz")
            try:
                effective = int(exported[-1])
            except (TypeError, IndexError):
                effective = int(exported)
        from ultralytics.utils.checks import check_imgsz
        predictor.imgsz = check_imgsz(
            effective,
            stride=backend.stride,
            min_dim=2,
        )
        predictor.args.imgsz = effective
        predictor.args.conf = self.conf
        predictor.args.iou = self.iou
        predictor.args.classes = self._class_filter
        return effective

    def _detections_from_results(self, results) -> list[Detection]:
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            data = getattr(boxes, "data", None)
            if data is None or int(getattr(data, "shape", [0])[0]) == 0:
                continue
            # One D2H transfer replaces three synchronizing scalar transfers
            # per bbox (cls, xyxy and confidence).
            rows = data[:, :6].detach().cpu().numpy()
            for x1, y1, x2, y2, confidence, class_id in rows:
                class_id_int = int(class_id)
                detections.append(Detection(
                    class_id=class_id_int,
                    class_name=self.model.names[class_id_int],
                    category=self._class_to_category.get(
                        class_id_int,
                        "unknown",
                    ),
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(confidence),
                ))
        return detections

    def _public_predict(
        self,
        frame: np.ndarray,
        img_size: int,
    ) -> tuple[list[Detection], DetectorTiming]:
        started = time.perf_counter()
        with self._torch.inference_mode():
            results = self.model.predict(
                frame,
                **self._predict_kwargs(img_size),
            )
        predict_ms = (time.perf_counter() - started) * 1000.0
        speeds = [getattr(result, "speed", None) or {} for result in results]
        divisor = max(1, len(speeds))
        preprocess_ms = sum(
            float(speed.get("preprocess", 0.0) or 0.0) for speed in speeds
        ) / divisor
        inference_ms = sum(
            float(speed.get("inference", 0.0) or 0.0) for speed in speeds
        ) / divisor
        library_postprocess_ms = sum(
            float(speed.get("postprocess", 0.0) or 0.0) for speed in speeds
        ) / divisor
        measured = preprocess_ms + inference_ms + library_postprocess_ms
        preprocess_ms += max(0.0, predict_ms - measured)
        conversion_started = time.perf_counter()
        detections = self._detections_from_results(results)
        conversion_ms = (time.perf_counter() - conversion_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        return detections, DetectorTiming(
            preprocess_ms=max(0.0, preprocess_ms),
            inference_ms=max(0.0, inference_ms),
            postprocess_ms=max(
                0.0,
                library_postprocess_ms + conversion_ms,
            ),
            total_ms=max(0.0, total_ms),
            img_size=int(img_size),
        )

    def _fast_predict(
        self,
        frame: np.ndarray,
        img_size: int,
    ) -> tuple[list[Detection], DetectorTiming]:
        predictor = self._predictor
        if predictor is None:
            raise RuntimeError("YOLO predictor is not warmed up")
        total_started = time.perf_counter()
        cuda_timing = self._is_cuda_device()
        with predictor._lock, self._torch.inference_mode():
            effective_size = self._set_predictor_size(img_size)
            predictor.batch = (["frame.jpg"], [frame], [""])
            preprocess_started = time.perf_counter()
            image = predictor.preprocess([frame])
            preprocess_ms = (
                time.perf_counter() - preprocess_started
            ) * 1000.0

            if cuda_timing:
                stream = self._torch.cuda.current_stream(image.device)
                inference_start = self._torch.cuda.Event(enable_timing=True)
                inference_end = self._torch.cuda.Event(enable_timing=True)
                postprocess_end = self._torch.cuda.Event(enable_timing=True)
                inference_start.record(stream)
            prediction = predictor.inference(image)
            if cuda_timing:
                inference_end.record(stream)
            else:
                inference_finished = time.perf_counter()

            results = predictor.postprocess(prediction, image, [frame])
            if cuda_timing:
                postprocess_end.record(stream)
            else:
                postprocess_finished = time.perf_counter()

            conversion_started = time.perf_counter()
            detections = self._detections_from_results(results)
            conversion_ms = (
                time.perf_counter() - conversion_started
            ) * 1000.0

            if cuda_timing:
                # A non-empty result has already waited through the single bulk
                # ``cpu()`` above.  An empty result has no D2H transfer at all,
                # so finish only its end event before reading elapsed times.
                # This replaces Ultralytics' six per-frame Profile syncs with
                # zero explicit syncs for detections and one for an empty frame.
                if not postprocess_end.query():
                    postprocess_end.synchronize()
                inference_ms = inference_start.elapsed_time(inference_end)
                gpu_postprocess_ms = inference_end.elapsed_time(postprocess_end)
            else:
                inference_ms = (
                    inference_finished
                    - (preprocess_started + preprocess_ms / 1000.0)
                ) * 1000.0
                gpu_postprocess_ms = (
                    postprocess_finished - inference_finished
                ) * 1000.0

        total_ms = (time.perf_counter() - total_started) * 1000.0
        accounted = preprocess_ms + inference_ms + gpu_postprocess_ms + conversion_ms
        preprocess_ms += max(0.0, total_ms - accounted)
        return detections, DetectorTiming(
            preprocess_ms=max(0.0, preprocess_ms),
            inference_ms=max(0.0, float(inference_ms)),
            postprocess_ms=max(
                0.0,
                float(gpu_postprocess_ms) + conversion_ms,
            ),
            total_ms=max(0.0, total_ms),
            img_size=effective_size,
        )

    def detect_profiled(
        self,
        frame: np.ndarray,
        *,
        img_size: int | None = None,
    ) -> tuple[list[Detection], DetectorTiming]:
        effective_size = int(img_size or self.img_size)
        if self._fast_path_active:
            try:
                return self._fast_predict(frame, effective_size)
            except Exception as exc:
                self._fast_path_active = False
                self._fast_path_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Optimized YOLO predictor disabled; using public API: %s",
                    exc,
                )
        return self._public_predict(frame, effective_size)

    def detect(
        self,
        frame: np.ndarray,
        *,
        img_size: int | None = None,
    ) -> list[Detection]:
        return self.detect_profiled(frame, img_size=img_size)[0]

    @property
    def supports_adaptive_size(self) -> bool:
        return bool(self._supports_adaptive_size)

    def status(self) -> dict:
        return {
            "backend_requested": self.requested_backend,
            "backend": self._actual_backend,
            "backend_fallback_used": self._fallback_used,
            "backend_notes": list(self._backend_notes),
            "model": self.model_path,
            "class_filter": list(self._class_filter),
            "class_categories": {
                str(class_id): self._class_to_category[class_id]
                for class_id in self._class_filter
            },
            "class_names": {
                str(class_id): str(self.model.names[class_id])
                for class_id in self._class_filter
                if class_id in self.model.names
            } if isinstance(self.model.names, dict) else {
                str(class_id): str(self.model.names[class_id])
                for class_id in self._class_filter
                if 0 <= class_id < len(self.model.names)
            },
            "device_requested": str(self.cfg.device),
            "device": str(self._actual_device),
            "precision_requested": self.requested_precision,
            "precision": self._actual_precision,
            "img_size": int(self.img_size),
            "adaptive_size_supported": self._supports_adaptive_size,
            "warmed_up": self._warmed_up,
            "warmup_ms": round(self._warmup_ms, 3),
            "fast_predictor": self._fast_path_active,
            "fast_predictor_error": self._fast_path_error,
            "onnx_runtime_cuda_available": self._onnx_cuda_available(),
            "tensorrt_available": self._tensorrt_available(),
        }


__all__ = [
    "Detection",
    "Detector",
    "DetectorTiming",
    "PPE_CATEGORIES",
    "SITE_CATEGORIES",
]
