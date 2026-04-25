from pathlib import Path
from typing import Any
from typing import Optional
from typing import cast

import numpy as np
from ultralytics import YOLO

try:
    from App.GestureRecon.detector import GestureAnalyzer
    from App.GestureRecon.hand_detector import HandDetector
except ImportError:
    try:
        from GestureRecon.detector import GestureAnalyzer
        from GestureRecon.hand_detector import HandDetector
    except ImportError:
        from .detector import GestureAnalyzer
        from .hand_detector import HandDetector

class GestureRecognitionService:
    def __init__(
        self,
        base_dir: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        fps: int = 12,
        tracker: str = "bytetrack.yaml",
        fallback_match_distance: float = 120.0,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.pose_model_path = Path(pose_model_path) if pose_model_path else (
            self.base_dir / "yolov8n-pose.pt"
        )
        self.tracker = tracker
        self.fallback_match_distance = fallback_match_distance

        if not self.pose_model_path.exists():
            raise FileNotFoundError(
                f"Modelo de pose nao encontrado: {self.pose_model_path}"
            )

        self.pose_model = YOLO(str(self.pose_model_path))
        self.analyzer = GestureAnalyzer(fps=fps)
        self.hand_detector = HandDetector()
        self.last_track_centers: dict[int, tuple[float, float]] = {}
        self.next_track_id = 1

    def detect_gestures(
        self,
        frame: np.ndarray,
        include_keypoints: bool = False,
    ) -> list[dict[str, Any]]:
        hand_detections = self.hand_detector.detect(frame)
        pose_results = self.pose_model.track(
            frame,
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
            hand_context = self._associate_hands(box, keypoints, hand_detections)
            analysis = self.analyzer.analyze(
                track_id,
                keypoints,
                box,
                hand_context=hand_context,
            )
            alerts = analysis["alerts"]

            people.append(
                {
                    "track_id": track_id,
                    "bbox": [int(value) for value in box],
                    "alerts": alerts,
                    "confidence": float(confidences[index]),
                    "hand_context": analysis["hand_context"],
                    "hidden_debug": analysis["hidden_debug"],
                    **(
                        {"matched_hands": hand_context["matched_hands"]}
                        if hand_context["matched_hands"]
                        else {}
                    ),
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
        include_keypoints: bool = False,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {"gestures": []}

        if detect_pose:
            response["gestures"] = self.detect_gestures(
                frame,
                include_keypoints=include_keypoints,
            )

        return response

    def _associate_hands(
        self,
        box: np.ndarray,
        keypoints: np.ndarray,
        hand_detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        x1, y1, x2, y2 = [float(value) for value in box]
        body_width = max(1.0, x2 - x1)
        body_height = max(1.0, y2 - y1)
        expanded_x1 = x1 - (body_width * 0.15)
        expanded_y1 = y1 - (body_height * 0.1)
        expanded_x2 = x2 + (body_width * 0.15)
        expanded_y2 = y2 + (body_height * 0.1)

        torso_box = self._build_torso_box(keypoints)
        left_wrist = self._keypoint_xy_conf(keypoints, 9)
        right_wrist = self._keypoint_xy_conf(keypoints, 10)

        context: dict[str, Any] = {
            "left_visible": False,
            "right_visible": False,
            "left_closed": False,
            "right_closed": False,
            "left_in_torso": False,
            "right_in_torso": False,
            "matched_hands": [],
        }

        left_candidate: Optional[dict[str, Any]] = None
        right_candidate: Optional[dict[str, Any]] = None

        for hand in hand_detections:
            center_x, center_y = hand["center"]
            if not (
                expanded_x1 <= center_x <= expanded_x2
                and expanded_y1 <= center_y <= expanded_y2
            ):
                continue

            side, distance = self._match_hand_side(hand, left_wrist, right_wrist)
            if side is None:
                continue

            hand_data = {
                "bbox": hand["bbox"],
                "center": hand["center"],
                "closed": hand["closed"],
                "in_torso": self._point_in_box(hand["center"], torso_box),
                "distance": distance,
            }

            if side == "left":
                if left_candidate is None or distance < left_candidate["distance"]:
                    left_candidate = hand_data
            else:
                if right_candidate is None or distance < right_candidate["distance"]:
                    right_candidate = hand_data

        if left_candidate is not None:
            context["left_visible"] = True
            context["left_closed"] = left_candidate["closed"]
            context["left_in_torso"] = left_candidate["in_torso"]
            context["matched_hands"].append(
                {
                    "side": "left",
                    "bbox": left_candidate["bbox"],
                    "closed": left_candidate["closed"],
                }
            )

        if right_candidate is not None:
            context["right_visible"] = True
            context["right_closed"] = right_candidate["closed"]
            context["right_in_torso"] = right_candidate["in_torso"]
            context["matched_hands"].append(
                {
                    "side": "right",
                    "bbox": right_candidate["bbox"],
                    "closed": right_candidate["closed"],
                }
            )

        return context

    def _to_numpy(self, data: Any) -> np.ndarray:
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "numpy"):
            return cast(np.ndarray, data.numpy())
        return np.asarray(data)

    def _keypoint_xy_conf(
        self,
        keypoints: np.ndarray,
        index: int,
    ) -> tuple[float, float, float]:
        keypoint = keypoints[index]
        return float(keypoint[0]), float(keypoint[1]), float(keypoint[2])

    def _point_in_box(
        self,
        point: tuple[float, float] | list[int],
        box: Optional[list[float]],
    ) -> bool:
        if box is None:
            return False

        point_x, point_y = point
        x1, y1, x2, y2 = box
        return x1 <= point_x <= x2 and y1 <= point_y <= y2

    def _build_torso_box(self, keypoints: np.ndarray) -> Optional[list[float]]:
        torso_indices = [5, 6, 11, 12]
        visible_points: list[tuple[float, float]] = []
        for index in torso_indices:
            x, y, conf = self._keypoint_xy_conf(keypoints, index)
            if conf > 0.4:
                visible_points.append((x, y))

        if len(visible_points) < 2:
            return None

        xs = [point[0] for point in visible_points]
        ys = [point[1] for point in visible_points]
        return [min(xs), min(ys), max(xs), max(ys)]

    def _match_hand_side(
        self,
        hand: dict[str, Any],
        left_wrist: tuple[float, float, float],
        right_wrist: tuple[float, float, float],
    ) -> tuple[Optional[str], Optional[float]]:
        best_side: Optional[str] = None
        best_distance: Optional[float] = None

        for side, wrist in (("left", left_wrist), ("right", right_wrist)):
            wrist_x, wrist_y, wrist_conf = wrist
            if wrist_conf <= 0.35:
                continue

            hand_x, hand_y = hand["center"]
            distance = float(
                ((hand_x - wrist_x) ** 2 + (hand_y - wrist_y) ** 2) ** 0.5
            )
            if best_distance is None or distance < best_distance:
                best_side = side
                best_distance = distance

        return best_side, best_distance

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
