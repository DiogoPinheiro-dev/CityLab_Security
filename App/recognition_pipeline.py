from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Optional

import numpy as np

try:
    from App.FaceRecon.service import FaceRecognitionService
    from App.GestureRecon.service import GestureRecognitionService
except ImportError:
    try:
        from FaceRecon.service import FaceRecognitionService
        from GestureRecon.service import GestureRecognitionService
    except ImportError:
        from .FaceRecon.service import FaceRecognitionService
        from .GestureRecon.service import GestureRecognitionService


class UnifiedRecognitionService:
    """
    Orquestra reconhecimento facial e de gestos sobre o mesmo frame.
    """

    def __init__(
        self,
        face_service: Optional[FaceRecognitionService] = None,
        gesture_service: Optional[GestureRecognitionService] = None,
        run_in_parallel: bool = True,
    ) -> None:
        self.face_service = face_service or FaceRecognitionService()
        self.gesture_service = gesture_service or GestureRecognitionService()
        self.run_in_parallel = run_in_parallel

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

        face_payload = self.face_service.process_frame(
            frame,
            detect_faces=detect_faces,
            detect_persons=detect_persons,
            with_logging=with_face_logging,
        )
        gesture_payload = self.gesture_service.process_frame(
            frame,
            detect_pose=detect_gestures,
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
                self.face_service.process_frame,
                frame,
                detect_faces,
                detect_persons,
                with_face_logging,
            )
            gesture_future = executor.submit(
                self.gesture_service.process_frame,
                frame,
                detect_gestures,
            )

            face_payload = face_future.result()
            gesture_payload = gesture_future.result()

        return self._merge_payloads(face_payload, gesture_payload)

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
