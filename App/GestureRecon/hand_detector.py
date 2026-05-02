import math
import os
import urllib.request
from typing import Any
from typing import Sequence
from typing import TypeAlias

import numpy as np

MP_MODULE: Any = None
MP_TASKS_PYTHON: Any = None
MP_VISION: Any = None
Point: TypeAlias = tuple[float, float]

try:
    import mediapipe as mediapipe_module
    from mediapipe.tasks import python as mediapipe_tasks_python
    from mediapipe.tasks.python import vision as mediapipe_vision
except ImportError:
    pass
else:
    MP_MODULE = mediapipe_module
    MP_TASKS_PYTHON = mediapipe_tasks_python
    MP_VISION = mediapipe_vision


class HandDetector:
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    )

    def __init__(
        self,
        max_num_hands: int = 4,
        min_detection_confidence: float = 0.35,
        min_tracking_confidence: float = 0.4,
    ) -> None:
        self.recognizer: Any = None
        self.available = all(
            module is not None
            for module in (MP_MODULE, MP_TASKS_PYTHON, MP_VISION)
        )
        self.model_path: str | None = None

        if not self.available:
            print("[AVISO] MediaPipe nao esta instalado. Detector de maos desativado.")
            return

        self.model_path = self._ensure_model()
        self._create_recognizer(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        if not self.available or self.recognizer is None:
            return []

        frame_h, frame_w = frame_bgr.shape[:2]
        image_rgb = frame_bgr[:, :, ::-1]
        mp_image = MP_MODULE.Image(
            image_format=MP_MODULE.ImageFormat.SRGB,
            data=image_rgb,
        )
        results = self.recognizer.recognize(mp_image)

        detected_hands: list[dict[str, Any]] = []
        landmarks_batch = results.hand_landmarks or []
        handedness_batch = results.handedness or []
        gestures_batch = results.gestures or []

        for index, landmarks in enumerate(landmarks_batch):
            points: list[Point] = []
            xs: list[int] = []
            ys: list[int] = []

            for landmark in landmarks:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                points.append((x, y))
                xs.append(x)
                ys.append(y)

            if not xs or not ys:
                continue

            bbox = [
                max(0, min(xs)),
                max(0, min(ys)),
                min(frame_w - 1, max(xs)),
                min(frame_h - 1, max(ys)),
            ]
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            handedness = "Unknown"
            handedness_score = 0.0
            if index < len(handedness_batch) and handedness_batch[index]:
                classification = handedness_batch[index][0]
                handedness = classification.category_name
                handedness_score = float(classification.score)

            gesture_name: str | None = None
            gesture_score = 0.0
            if index < len(gestures_batch) and gestures_batch[index]:
                gesture = gestures_batch[index][0]
                gesture_name = gesture.category_name
                gesture_score = float(gesture.score)

            closed = (
                gesture_name == "Closed_Fist"
                or self._is_closed_fist(points)
                or self._is_side_closed_fist(points)
            )

            detected_hands.append(
                {
                    "bbox": bbox,
                    "center": [center_x, center_y],
                    "landmarks": points,
                    "handedness": handedness,
                    "score": handedness_score,
                    "gesture_name": gesture_name,
                    "gesture_score": gesture_score,
                    "closed": closed,
                }
            )

        return detected_hands

    def _create_recognizer(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        if (
            not self.available
            or not self.model_path
            or not os.path.exists(self.model_path)
        ):
            return

        base_options = MP_TASKS_PYTHON.BaseOptions(model_asset_path=self.model_path)
        options = MP_VISION.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=MP_VISION.RunningMode.IMAGE,
        )
        self.recognizer = MP_VISION.GestureRecognizer.create_from_options(options)

    def _ensure_model(self) -> str | None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "gesture_recognizer.task")
        if os.path.exists(model_path):
            return model_path

        try:
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            return model_path
        except Exception as exc:
            print(f"[AVISO] Nao foi possivel baixar o modelo do MediaPipe: {exc}")
            return None

    def _distance(self, point_a: Point, point_b: Point) -> float:
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    def _is_closed_fist(self, points: Sequence[Point]) -> bool:
        if len(points) < 21:
            return False

        wrist = points[0]
        index_mcp = points[5]
        pinky_mcp = points[17]

        palm_size = max(
            1.0,
            (
                self._distance(wrist, index_mcp)
                + self._distance(wrist, pinky_mcp)
                + self._distance(index_mcp, pinky_mcp)
            )
            / 3.0,
        )

        curled_fingers = 0
        finger_triplets = [
            (8, 6, 5),
            (12, 10, 9),
            (16, 14, 13),
            (20, 18, 17),
        ]
        for tip_idx, pip_idx, mcp_idx in finger_triplets:
            tip = points[tip_idx]
            pip = points[pip_idx]
            mcp = points[mcp_idx]

            tip_to_wrist = self._distance(tip, wrist)
            pip_to_wrist = self._distance(pip, wrist)
            tip_to_mcp = self._distance(tip, mcp)

            if tip_to_wrist < (pip_to_wrist * 1.05) and tip_to_mcp < (palm_size * 0.9):
                curled_fingers += 1

        thumb_tip = points[4]
        thumb_ip = points[3]
        thumb_mcp = points[2]
        thumb_curled = self._distance(thumb_tip, thumb_mcp) < self._distance(
            thumb_ip, thumb_mcp
        ) * 1.15

        return curled_fingers >= 3 and thumb_curled

    def _is_side_closed_fist(self, points: Sequence[Point]) -> bool:
        if len(points) < 21:
            return False

        wrist = points[0]
        index_mcp = points[5]
        middle_mcp = points[9]
        pinky_mcp = points[17]
        fingertips = [points[index] for index in (8, 12, 16, 20)]
        palm_center = (
            (index_mcp[0] + middle_mcp[0] + pinky_mcp[0]) / 3.0,
            (index_mcp[1] + middle_mcp[1] + pinky_mcp[1]) / 3.0,
        )

        palm_size = max(
            1.0,
            (
                self._distance(wrist, index_mcp)
                + self._distance(wrist, pinky_mcp)
                + self._distance(index_mcp, pinky_mcp)
            )
            / 3.0,
        )

        compact_tips = 0
        fingertip_distances = []
        for tip, mcp in zip(fingertips, (points[5], points[9], points[13], points[17])):
            tip_to_center = self._distance(tip, palm_center)
            tip_to_mcp = self._distance(tip, mcp)
            fingertip_distances.append(tip_to_center)
            if tip_to_center < (palm_size * 1.15) and tip_to_mcp < (palm_size * 1.2):
                compact_tips += 1

        fingertip_spread = max(
            self._distance(fingertips[i], fingertips[j])
            for i in range(len(fingertips))
            for j in range(i + 1, len(fingertips))
        )
        avg_tip_to_center = sum(fingertip_distances) / len(fingertip_distances)

        thumb_tip = points[4]
        thumb_mcp = points[2]
        thumb_to_palm = self._distance(thumb_tip, palm_center)
        thumb_compact = (
            thumb_to_palm < (palm_size * 1.35)
            or self._distance(thumb_tip, thumb_mcp) < (palm_size * 0.95)
        )

        return (
            compact_tips >= 3
            and avg_tip_to_center < (palm_size * 1.05)
            and fingertip_spread < (palm_size * 1.35)
            and thumb_compact
        )
