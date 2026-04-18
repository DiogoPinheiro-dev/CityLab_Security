from pathlib import Path
from typing import Any
from typing import Optional
from typing import cast

import numpy as np
from ultralytics import YOLO

try:
    from App.GestureRecon.detector import GestureAnalyzer
    from App.camera_auto_config import AutoImageOptimizer
except ImportError:
    try:
        from GestureRecon.detector import GestureAnalyzer
        from camera_auto_config import AutoImageOptimizer
    except ImportError:
        from .detector import GestureAnalyzer
        from ..camera_auto_config import AutoImageOptimizer


class GestureRecognitionService:
    def __init__(
        self,
        base_dir: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        object_model_path: Optional[str] = None,
        fps: int = 12,
        tracker: str = "bytetrack.yaml",
        target_classes: Optional[list[int]] = None,
        object_confidence_threshold: float = 0.4,
        fallback_match_distance: float = 120.0,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.pose_model_path = Path(pose_model_path) if pose_model_path else (
            self.base_dir / "yolov8n-pose.pt"
        )
        self.object_model_path = (
            Path(object_model_path)
            if object_model_path
            else self._find_default_object_model()
        )
        self.tracker = tracker
        self.target_classes = target_classes or [43, 67]
        self.object_confidence_threshold = object_confidence_threshold
        self.fallback_match_distance = fallback_match_distance

        if not self.pose_model_path.exists():
            raise FileNotFoundError(
                f"Modelo de pose nao encontrado: {self.pose_model_path}"
            )

        if self.object_model_path is None or not self.object_model_path.exists():
            raise FileNotFoundError(
                f"Modelo de objetos nao encontrado: {self.object_model_path}"
            )

        self.pose_model = YOLO(str(self.pose_model_path))
        self.object_model = YOLO(str(self.object_model_path))
        self.analyzer = GestureAnalyzer(fps=fps)
        self.image_optimizer = AutoImageOptimizer()
        self.last_track_centers: dict[int, tuple[float, float]] = {}
        self.next_track_id = 1

    def detect_objects(
        self,
        frame: np.ndarray,
        optimized_frame: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        processed_frame = (
            optimized_frame
            if optimized_frame is not None
            else self.image_optimizer.optimize(frame)
        )
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
            cls_id = int(self._scalar_from_tensor(box.cls[0]))
            conf = float(self._scalar_from_tensor(box.conf[0]))
            if conf < self.object_confidence_threshold:
                continue

            x1, y1, x2, y2 = self._bbox_from_tensor(box.xyxy[0])
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
        include_keypoints: bool = False,
    ) -> list[dict[str, Any]]:
        processed_frame = (
            optimized_frame
            if optimized_frame is not None
            else self.image_optimizer.optimize(frame)
        )
        pose_results = self.pose_model.track(
            processed_frame,
            persist=True,
            tracker=self.tracker,
            classes=[0],
            verbose=False,
        )

        people: list[dict[str, Any]] = []
        current_tracks: list[int] = []

        if not pose_results:
            self.analyzer.clean_old_tracks(current_tracks)
            return people

        result = pose_results[0]
        if result.boxes is None or result.keypoints is None:
            self.analyzer.clean_old_tracks(current_tracks)
            return people

        boxes = self._to_numpy(result.boxes.xyxy)
        keypoints_batch = self._to_numpy(result.keypoints.data)
        confidences = self._to_numpy(result.boxes.conf)
        track_ids = self._resolve_track_ids(
            boxes,
            result.boxes.id,
        )

        for index, (box, track_id, keypoints) in enumerate(
            zip(boxes, track_ids, keypoints_batch)
        ):
            current_tracks.append(track_id)
            alerts = self.analyzer.analyze(track_id, keypoints, box)

            people.append(
                {
                    "track_id": track_id,
                    "bbox": [int(value) for value in box],
                    "alerts": alerts,
                    "confidence": float(confidences[index]),
                    **(
                        {"keypoints": keypoints.tolist()}
                        if include_keypoints
                        else {}
                    ),
                }
            )

        self.analyzer.clean_old_tracks(current_tracks)
        return people

    def process_frame(
        self,
        frame: np.ndarray,
        detect_pose: bool = True,
        detect_objects: bool = True,
        include_keypoints: bool = False,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {"gestures": [], "objects": []}
        optimized_frame = self.image_optimizer.optimize(frame)

        if detect_pose:
            response["gestures"] = self.detect_gestures(
                frame,
                optimized_frame=optimized_frame,
                include_keypoints=include_keypoints,
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

    def _find_default_object_model(self) -> Optional[Path]:
        candidate_paths = [
            self.base_dir / "yolov8n.pt",
            self.base_dir.parent / "FaceRecon" / "yolov8n.pt",
        ]

        for candidate in candidate_paths:
            if candidate.exists():
                return candidate

        return None

    def _to_numpy(self, data: Any) -> np.ndarray:
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "numpy"):
            return cast(np.ndarray, data.numpy())
        return np.asarray(data)

    def _scalar_from_tensor(self, value: Any) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _bbox_from_tensor(self, bbox: Any) -> tuple[int, int, int, int]:
        values = self._to_numpy(bbox).tolist()
        x1, y1, x2, y2 = [int(value) for value in values]
        return x1, y1, x2, y2

    def _resolve_track_ids(
        self,
        boxes: np.ndarray,
        raw_track_ids: Any,
    ) -> list[int]:
        if raw_track_ids is not None:
            track_ids = [int(track_id) for track_id in self._to_numpy(raw_track_ids)]
            self.last_track_centers = {
                track_id: self._box_center(box)
                for track_id, box in zip(track_ids, boxes)
            }
            if track_ids:
                self.next_track_id = max(self.next_track_id, max(track_ids) + 1)
            return track_ids

        assigned_track_ids: list[int] = []
        remaining_previous = dict(self.last_track_centers)
        updated_centers: dict[int, tuple[float, float]] = {}

        for box in boxes:
            center = self._box_center(box)
            best_track_id: Optional[int] = None
            best_distance = float("inf")

            for track_id, previous_center in remaining_previous.items():
                distance = self._center_distance(center, previous_center)
                if distance < best_distance and distance <= self.fallback_match_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self.next_track_id
                self.next_track_id += 1
            else:
                remaining_previous.pop(best_track_id, None)

            assigned_track_ids.append(best_track_id)
            updated_centers[best_track_id] = center

        self.last_track_centers = updated_centers
        return assigned_track_ids

    def _box_center(self, box: np.ndarray) -> tuple[float, float]:
        x1, y1, x2, y2 = [float(value) for value in box]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _center_distance(
        self,
        center_a: tuple[float, float],
        center_b: tuple[float, float],
    ) -> float:
        delta_x = center_a[0] - center_b[0]
        delta_y = center_a[1] - center_b[1]
        return float((delta_x ** 2 + delta_y ** 2) ** 0.5)


def create_gesture_service(**kwargs: Any) -> GestureRecognitionService:
    return GestureRecognitionService(**kwargs)
