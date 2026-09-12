import os
import pickle
from typing import Any
from typing import Optional

import insightface
import numpy as np
from ultralytics import YOLO  # type: ignore

from App.frame_context import FrameContext
from App.settings import DEBUG_PIPELINE, FACE_MIN_CONFIDENCE, FACE_MIN_HEIGHT, FACE_MIN_WIDTH


class FaceRecognitionService:
    def __init__(
        self,
        base_dir: Optional[str] = None,
        database_path: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
        similarity_threshold: float = 0.52,
        face_model_name: str = "buffalo_l",
        insight_providers: Optional[list[str]] = None,
        insight_det_size: tuple[int, int] = (320, 320),
        face_min_width: int = FACE_MIN_WIDTH,
        face_min_height: int = FACE_MIN_HEIGHT,
        face_min_confidence: float = FACE_MIN_CONFIDENCE,
        debug_pipeline: bool = DEBUG_PIPELINE,
    ) -> None:
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.database_path = database_path or os.path.join(
            self.base_dir, "base_dados_alunos.pkl"
        )
        self.yolo_model_path = yolo_model_path or os.path.join(
            self.base_dir, "yolov8n.pt"
        )

        if not os.path.exists(self.yolo_model_path):
            raise FileNotFoundError(
                f"Modelo YOLO nao encontrado: {self.yolo_model_path}"
            )

        self.similarity_threshold = similarity_threshold
        self.face_min_width = face_min_width
        self.face_min_height = face_min_height
        self.face_min_confidence = face_min_confidence
        self.debug_pipeline = debug_pipeline

        self.known_face_embeddings = np.empty((0, 512), dtype=np.float32)
        self.known_face_names: list[str] = []
        self._load_database()

        self.model_yolo = YOLO(self.yolo_model_path)
        self.app_insight = insightface.app.FaceAnalysis(
            name=face_model_name,
            providers=insight_providers or ["CPUExecutionProvider"],
        )
        self.app_insight.prepare(ctx_id=0, det_size=insight_det_size)
        self.latest_metrics: dict[str, float] = {
            "faces_ms": 0.0,
            "persons_ms": 0.0,
        }
        self.latest_ignored_faces: list[dict[str, Any]] = []

    def _load_database(self) -> None:
        if not os.path.exists(self.database_path):
            return

        with open(self.database_path, "rb") as file:
            data = pickle.load(file)

        embeddings = data.get("embeddings", [])
        names = data.get("names", [])

        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        if embeddings_array.size > 0:
            self.known_face_embeddings = embeddings_array
        self.known_face_names = list(names)

    def replace_known_faces(
        self, names: list[str], embeddings: list[np.ndarray] | np.ndarray
    ) -> None:
        self.known_face_names = list(names)
        if len(embeddings) == 0:
            self.known_face_embeddings = np.empty((0, 512), dtype=np.float32)
            return

        self.known_face_embeddings = np.asarray(embeddings, dtype=np.float32)

    def recognize_faces(
        self,
        frame_context: FrameContext,
    ) -> list[dict[str, Any]]:
        import time

        started_at = time.perf_counter()
        faces = self.app_insight.get(frame_context.processing_frame)
        results: list[dict[str, Any]] = []
        ignored_faces: list[dict[str, Any]] = []

        for face in faces:
            bbox_processing = face.bbox.astype(int).tolist()
            bbox_original = frame_context.clip_original_bbox(
                frame_context.map_bbox_to_original(bbox_processing)
            )
            quality_reason = self._validate_face(face, bbox_original)
            if quality_reason is not None:
                if self.debug_pipeline:
                    ignored_faces.append(
                        {
                            "bbox": bbox_original,
                            "det_confidence": float(getattr(face, "det_score", 0.0)),
                            "reason": quality_reason,
                        }
                    )
                continue

            name, best_score = self._match_face(face.normed_embedding)

            results.append(
                {
                    "name": name,
                    "bbox": bbox_original,
                    "confidence": float(best_score),
                    "det_confidence": float(getattr(face, "det_score", 0.0)),
                }
            )

        if self.debug_pipeline and ignored_faces:
            self.latest_metrics["ignored_faces"] = float(len(ignored_faces))
        self.latest_ignored_faces = ignored_faces
        self.latest_metrics["faces_ms"] = (time.perf_counter() - started_at) * 1000.0
        return results

    def detect_persons(
        self,
        frame_context: FrameContext,
    ) -> list[dict[str, Any]]:
        import time

        started_at = time.perf_counter()
        results_yolo = self.model_yolo(
            frame_context.processing_frame,
            classes=[0],
            verbose=False,
        )

        persons: list[dict[str, Any]] = []
        for result in results_yolo:
            for box in result.boxes:
                bbox_person = box.xyxy[0].numpy().astype(int)
                persons.append(
                    {
                        "bbox": frame_context.clip_original_bbox(
                            frame_context.map_bbox_to_original(bbox_person.tolist())
                        ),
                        "confidence": float(box.conf[0]),
                    }
                )
        self.latest_metrics["persons_ms"] = (time.perf_counter() - started_at) * 1000.0
        return persons

    def process_frame(
        self,
        frame_context: FrameContext,
        detect_faces: bool = True,
        detect_persons: bool = True,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {"faces": [], "persons": []}

        if detect_faces:
            response["faces"] = self.recognize_faces(frame_context)

        if detect_persons:
            response["persons"] = self.detect_persons(frame_context)

        return response

    def _match_face(self, live_embedding: np.ndarray) -> tuple[str, float]:
        if len(self.known_face_embeddings) == 0:
            return "NAO ALUNO", 0.0

        scores = np.dot(self.known_face_embeddings, live_embedding)
        best_match_index = int(np.argmax(scores))
        best_score = float(scores[best_match_index])

        if best_score > self.similarity_threshold:
            return self.known_face_names[best_match_index], best_score

        return "NAO ALUNO", best_score

    def _validate_face(self, face: Any, bbox_original: list[int]) -> str | None:
        x1, y1, x2, y2 = bbox_original
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        det_confidence = float(getattr(face, "det_score", 0.0))

        if width <= 0 or height <= 0:
            return "bbox_invalida"
        if width < self.face_min_width or height < self.face_min_height:
            return "baixa_qualidade"
        if det_confidence < self.face_min_confidence:
            return "baixa_confianca"
        return None


def create_face_service(**kwargs: Any) -> FaceRecognitionService:
    return FaceRecognitionService(**kwargs)
