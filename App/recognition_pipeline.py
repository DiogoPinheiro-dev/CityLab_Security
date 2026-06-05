import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

import numpy as np

from App.frame_context import build_frame_context
from App.settings import (
    DEBUG_PIPELINE,
    ENABLE_PERFORMANCE_METRICS,
    EXPERIMENTAL_GRAYSCALE,
    PIPELINE_MAX_WORKERS,
    PIPELINE_RUN_IN_PARALLEL,
    PROCESS_SCALE,
)

if TYPE_CHECKING:
    from App.FaceRecon.service import FaceRecognitionService
    from App.GestureRecon.service import GestureRecognitionService


logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _load_face_service_class() -> type["FaceRecognitionService"]:
    try:
        from App.FaceRecon.service import FaceRecognitionService
    except ImportError:
        try:
            from FaceRecon.service import FaceRecognitionService
        except ImportError:
            from .FaceRecon.service import FaceRecognitionService

    return FaceRecognitionService


def _load_gesture_service_class() -> type["GestureRecognitionService"]:
    try:
        from App.GestureRecon.service import GestureRecognitionService
    except ImportError:
        try:
            from GestureRecon.service import GestureRecognitionService
        except ImportError:
            from .GestureRecon.service import GestureRecognitionService

    return GestureRecognitionService


class UnifiedRecognitionService:
    """
    Orquestra reconhecimento facial e de gestos sobre o mesmo frame.
    """

    def __init__(
        self,
        face_service: Optional["FaceRecognitionService"] = None,
        gesture_service: Optional["GestureRecognitionService"] = None,
        run_in_parallel: bool = PIPELINE_RUN_IN_PARALLEL,
        max_workers: int = PIPELINE_MAX_WORKERS,
        process_scale: float = PROCESS_SCALE,
        experimental_grayscale: bool = EXPERIMENTAL_GRAYSCALE,
        debug_pipeline: bool = DEBUG_PIPELINE,
        enable_performance_metrics: bool = ENABLE_PERFORMANCE_METRICS,
    ) -> None:
        self.allow_partial_pipeline = _env_bool("CITYLAB_ALLOW_PARTIAL_PIPELINE", False)
        self.face_service = face_service or self._create_face_service()
        self.gesture_service = gesture_service or self._create_gesture_service()
        self.run_in_parallel = run_in_parallel
        self.process_scale = process_scale
        self.experimental_grayscale = experimental_grayscale
        self.debug_pipeline = debug_pipeline
        self.enable_performance_metrics = enable_performance_metrics or debug_pipeline
        self.executor: Optional[ThreadPoolExecutor] = None
        self.max_workers = max(1, max_workers)
        self._last_frame_ended_at: Optional[float] = None

        if self.run_in_parallel:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def _create_face_service(self) -> Optional["FaceRecognitionService"]:
        if not _env_bool("CITYLAB_ENABLE_FACE_SERVICE", True):
            logger.warning("Face service desativado por CITYLAB_ENABLE_FACE_SERVICE.")
            return None

        try:
            return _load_face_service_class()()
        except Exception as exc:
            if not self.allow_partial_pipeline:
                raise

            logger.warning("Face service indisponivel: %s", exc)
            return None

    def _create_gesture_service(self) -> Optional["GestureRecognitionService"]:
        if not _env_bool("CITYLAB_ENABLE_GESTURE_SERVICE", True):
            logger.warning(
                "Gesture service desativado por CITYLAB_ENABLE_GESTURE_SERVICE."
            )
            return None

        try:
            return _load_gesture_service_class()()
        except Exception as exc:
            if not self.allow_partial_pipeline:
                raise

            logger.warning("Gesture service indisponivel: %s", exc)
            return None

    def process_frame(
        self,
        frame: np.ndarray,
        detect_faces: bool = True,
        detect_persons: bool = True,
        detect_gestures: bool = True,
    ) -> dict[str, Any]:
        import time

        started_at = time.perf_counter()
        metrics: dict[str, float] = {}

        frame_context = build_frame_context(
            frame,
            process_scale=self.process_scale,
            experimental_grayscale=self.experimental_grayscale,
        )

        persons: list[dict[str, Any]] = []
        if detect_persons and self.face_service is not None:
            persons = self.face_service.detect_persons(frame_context)
            metrics["persons_ms"] = self.face_service.latest_metrics.get("persons_ms", 0.0)
        else:
            metrics["persons_ms"] = 0.0

        faces: list[dict[str, Any]] = []
        gestures: list[dict[str, Any]] = []

        if self.run_in_parallel and self.executor is not None:
            faces, gestures = self._process_parallel(
                frame_context=frame_context,
                persons=persons,
                detect_faces=detect_faces,
                detect_gestures=detect_gestures,
                metrics=metrics,
            )
        else:
            if detect_faces and self.face_service is not None:
                faces = self.face_service.recognize_faces(frame_context)
                metrics["faces_ms"] = self.face_service.latest_metrics.get("faces_ms", 0.0)
            else:
                metrics["faces_ms"] = 0.0

            if detect_gestures and persons and self.gesture_service is not None:
                gestures = self.gesture_service.detect_gestures(
                    frame_context,
                    person_bboxes=persons,
                )
                metrics["gestures_ms"] = self.gesture_service.latest_metrics.get("gestures_ms", 0.0)
                metrics["hands_ms"] = self.gesture_service.latest_metrics.get("hands_ms", 0.0)
                metrics["pose_ms"] = self.gesture_service.latest_metrics.get("pose_ms", 0.0)
            else:
                metrics["gestures_ms"] = 0.0
                metrics["hands_ms"] = 0.0
                metrics["pose_ms"] = 0.0

        payload = self._merge_payloads(
            face_payload={"faces": faces, "persons": persons},
            gesture_payload={"gestures": gestures},
        )

        if self.enable_performance_metrics:
            finished_at = time.perf_counter()
            total_ms = (finished_at - started_at) * 1000.0
            metrics["total_ms"] = total_ms
            metrics["effective_fps"] = 0.0 if total_ms <= 0 else 1000.0 / total_ms
            if self._last_frame_ended_at is not None:
                metrics["frame_interval_ms"] = (
                    finished_at - self._last_frame_ended_at
                ) * 1000.0
            self._last_frame_ended_at = finished_at
            payload["metrics"] = metrics

        if self.debug_pipeline:
            payload["debug"] = {
                "process_scale": self.process_scale,
                "processing_resolution": [
                    int(frame_context.processing_frame.shape[1]),
                    int(frame_context.processing_frame.shape[0]),
                ],
                "original_resolution": [
                    int(frame_context.original_frame.shape[1]),
                    int(frame_context.original_frame.shape[0]),
                ],
                "ignored_faces": (
                    self.face_service.latest_ignored_faces
                    if self.face_service is not None
                    else []
                ),
            }
        return payload

    def _process_parallel(
        self,
        frame_context: Any,
        persons: list[dict[str, Any]],
        detect_faces: bool,
        detect_gestures: bool,
        metrics: dict[str, float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        face_future = None
        gesture_future = None

        if detect_faces and self.face_service is not None and self.executor is not None:
            face_future = self.executor.submit(
                self.face_service.recognize_faces,
                frame_context,
            )
        if (
            detect_gestures
            and persons
            and self.gesture_service is not None
            and self.executor is not None
        ):
            gesture_future = self.executor.submit(
                self.gesture_service.detect_gestures,
                frame_context,
                persons,
            )

        faces: list[dict[str, Any]] = []
        gestures: list[dict[str, Any]] = []

        if face_future is not None and self.face_service is not None:
            faces = face_future.result()
            metrics["faces_ms"] = self.face_service.latest_metrics.get("faces_ms", 0.0)
        else:
            metrics["faces_ms"] = 0.0

        if gesture_future is not None and self.gesture_service is not None:
            gestures = gesture_future.result()
            metrics["gestures_ms"] = self.gesture_service.latest_metrics.get("gestures_ms", 0.0)
            metrics["hands_ms"] = self.gesture_service.latest_metrics.get("hands_ms", 0.0)
            metrics["pose_ms"] = self.gesture_service.latest_metrics.get("pose_ms", 0.0)
        else:
            metrics["gestures_ms"] = 0.0
            metrics["hands_ms"] = 0.0
            metrics["pose_ms"] = 0.0

        return faces, gestures

    def _merge_payloads(
        self,
        face_payload: dict[str, Any],
        gesture_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "faces": face_payload.get("faces", []),
            "persons": face_payload.get("persons", []),
            "gestures": gesture_payload.get("gestures", []),
        }

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None


def create_unified_service(**kwargs: Any) -> UnifiedRecognitionService:
    return UnifiedRecognitionService(**kwargs)
