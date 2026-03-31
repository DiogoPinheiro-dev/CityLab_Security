import os
from typing import Any
from typing import Optional

import numpy as np
from ultralytics import YOLO

from GestureRecon.detector import GestureAnalyzer
from camera_auto_config import AutoImageOptimizer


class GestureRecognitionService:
    def __init__(
        self,
        base_dir: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        object_model_path: Optional[str] = None,
        fps: int = 30,
        tracker: str = "bytetrack.yaml",
        target_classes: Optional[list[int]] = None,
        object_confidence_threshold: float = 0.4,
    ) -> None:
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.pose_model_path = pose_model_path or os.path.join(
            self.base_dir, "yolov8n-pose.pt"
        )
        self.object_model_path = object_model_path or self._find_default_object_model()
        self.tracker = tracker
        self.target_classes = target_classes or [43, 67]
        self.object_confidence_threshold = object_confidence_threshold

        if not os.path.exists(self.pose_model_path):
            raise FileNotFoundError(
                f"Modelo de pose nao encontrado: {self.pose_model_path}"
            )

        if not self.object_model_path or not os.path.exists(self.object_model_path):
            raise FileNotFoundError(
                f"Modelo de objetos nao encontrado: {self.object_model_path}"
            )

        self.pose_model = YOLO(self.pose_model_path)
        self.object_model = YOLO(self.object_model_path)
        self.analyzer = GestureAnalyzer(fps=fps)
        self.image_optimizer = AutoImageOptimizer()

    def detect_objects(
        self,
        frame: np.ndarray,
        optimized_frame: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        processed_frame = optimized_frame if optimized_frame is not None else self.image_optimizer.optimize(frame)
        obj_results = self.object_model(
            processed_frame,
            classes=self.target_classes,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        if not obj_results:
            return detections

        result = obj_results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if conf < self.object_confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            detections.append(
                {
                    "class_id": cls_id,
                    "label": self._resolve_object_label(cls_id),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                }
            )

        return detections

    def detect_gestures(
        self,
        frame: np.ndarray,
        optimized_frame: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        processed_frame = optimized_frame if optimized_frame is not None else self.image_optimizer.optimize(frame)
        pose_results = self.pose_model.track(
            processed_frame,
            persist=True,
            tracker=self.tracker,
            verbose=False,
        )

        people: list[dict[str, Any]] = []
        current_tracks: list[int] = []

        if not pose_results:
            self.analyzer.clean_old_tracks(current_tracks)
            return people

        result = pose_results[0]
        if (
            result.boxes is None
            or result.boxes.id is None
            or result.keypoints is None
        ):
            self.analyzer.clean_old_tracks(current_tracks)
            return people

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        keypoints_batch = result.keypoints.data.cpu().numpy()

        for box, track_id, keypoints in zip(boxes, track_ids, keypoints_batch):
            current_tracks.append(track_id)
            alerts = self.analyzer.analyze(track_id, keypoints, box)

            people.append(
                {
                    "track_id": track_id,
                    "bbox": [int(value) for value in box],
                    "alerts": alerts,
                    "keypoints": keypoints.tolist(),
                }
            )

        self.analyzer.clean_old_tracks(current_tracks)
        return people

    def process_frame(
        self,
        frame: np.ndarray,
        detect_pose: bool = True,
        detect_objects: bool = True,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {"gestures": [], "objects": []}
        optimized_frame = self.image_optimizer.optimize(frame)

        if detect_pose:
            response["gestures"] = self.detect_gestures(
                frame,
                optimized_frame=optimized_frame,
            )

        if detect_objects:
            response["objects"] = self.detect_objects(
                frame,
                optimized_frame=optimized_frame,
            )

        return response

    def _resolve_object_label(self, cls_id: int) -> str:
        if cls_id == 67:
            return "Objeto Suspeito (Celular/Arma Fake)"
        if cls_id == 43:
            return "Arma Branca (Faca)"
        return f"Classe {cls_id}"

    def _find_default_object_model(self) -> Optional[str]:
        candidate_paths = [
            os.path.join(self.base_dir, "yolov8n.pt"),
            os.path.join(os.path.dirname(self.base_dir), "FaceRecon", "yolov8n.pt"),
        ]

        for candidate in candidate_paths:
            if os.path.exists(candidate):
                return candidate

        return None


def create_gesture_service(**kwargs: Any) -> GestureRecognitionService:
    return GestureRecognitionService(**kwargs)
