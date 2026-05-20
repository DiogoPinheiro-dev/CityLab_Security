import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

import numpy as np

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
        run_in_parallel: bool = True,
    ) -> None:
        self.allow_partial_pipeline = _env_bool("CITYLAB_ALLOW_PARTIAL_PIPELINE", False)
        self.face_service = face_service or self._create_face_service()
        self.gesture_service = gesture_service or self._create_gesture_service()
        self.run_in_parallel = run_in_parallel

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
        with_face_logging: bool = True,
    ) -> dict[str, Any]:
        if self.run_in_parallel:
            return self._process_parallel(
                frame=frame,
                detect_faces=detect_faces,
                detect_persons=detect_persons,
                detect_gestures=detect_gestures,
                with_face_logging=with_face_logging,
            )

        face_payload = self._process_face(
            frame=frame,
            detect_faces=detect_faces,
            detect_persons=detect_persons,
            with_face_logging=with_face_logging,
        )
        gesture_payload = self._process_gesture(
            frame=frame,
            detect_gestures=detect_gestures,
        )
        return self._merge_payloads(face_payload, gesture_payload)

    def _process_parallel(
        self,
        frame: np.ndarray,
        detect_faces: bool,
        detect_persons: bool,
        detect_gestures: bool,
        with_face_logging: bool,
    ) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            face_future = executor.submit(
                self._process_face,
                frame,
                detect_faces,
                detect_persons,
                with_face_logging,
            )
            gesture_future = executor.submit(
                self._process_gesture,
                frame,
                detect_gestures,
            )

            face_payload = face_future.result()
            gesture_payload = gesture_future.result()

        return self._merge_payloads(face_payload, gesture_payload)

    def _process_face(
        self,
        frame: np.ndarray,
        detect_faces: bool,
        detect_persons: bool,
        with_face_logging: bool,
    ) -> dict[str, Any]:
        if self.face_service is None:
            return {"faces": [], "persons": []}

        return self.face_service.process_frame(
            frame,
            detect_faces=detect_faces,
            detect_persons=detect_persons,
            with_logging=with_face_logging,
        )

    def _process_gesture(
        self,
        frame: np.ndarray,
        detect_gestures: bool,
    ) -> dict[str, Any]:
        if self.gesture_service is None:
            return {"gestures": []}

        return self.gesture_service.process_frame(
            frame,
            detect_pose=detect_gestures,
        )

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


def create_unified_service(**kwargs: Any) -> UnifiedRecognitionService:
    return UnifiedRecognitionService(**kwargs)
