import os
import pickle
from typing import Any
from typing import Optional

import insightface
from insightface.app.common import Face
import numpy as np
from ultralytics import YOLO  # type: ignore

from App.frame_context import FrameContext
from App.settings import (DEBUG_PIPELINE, FACE_MIN_CONFIDENCE, FACE_MIN_HEIGHT,
                          FACE_MIN_WIDTH, FACE_MINIMAL_MODULES, FACE_PREFILTER,
                          FACE_REUSE_SECONDS, ONNX_INTRA_OP_THREADS)
from App.inference_runtime import configure_insight_threads


class FaceRecognitionService:
    # Sobreposicao minima entre caixas de rosto de frames seguidos para tratar
    # como o mesmo rosto.
    REUSE_MIN_IOU = 0.5

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
        lazy_person_model: bool = False,
        minimal_modules: bool = FACE_MINIMAL_MODULES,
        prefilter: bool = FACE_PREFILTER,
        onnx_threads: int = ONNX_INTRA_OP_THREADS,
        reuse_seconds: float = FACE_REUSE_SECONDS,
    ) -> None:
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.database_path = database_path or os.path.join(
            self.base_dir, "base_dados_alunos.pkl"
        )
        self.yolo_model_path = yolo_model_path or os.path.join(
            self.base_dir, "yolov8n.pt"
        )

        if not lazy_person_model and not os.path.exists(self.yolo_model_path):
            raise FileNotFoundError(
                f"Modelo YOLO nao encontrado: {self.yolo_model_path}"
            )

        self.similarity_threshold = similarity_threshold
        self.face_min_width = face_min_width
        self.face_min_height = face_min_height
        self.face_min_confidence = face_min_confidence
        self.debug_pipeline = debug_pipeline
        self.prefilter = prefilter
        # Zero desliga: todo rosto aceito gera embedding em todo frame.
        self.reuse_seconds = reuse_seconds
        self._identities: list[dict[str, Any]] = []
        self._identity_generation = 0

        self.known_face_embeddings = np.empty((0, 512), dtype=np.float32)
        self.known_face_names: list[str] = []
        self._load_database()

        self.model_yolo = None if lazy_person_model else YOLO(self.yolo_model_path)
        self.app_insight = insightface.app.FaceAnalysis(
            name=face_model_name,
            providers=insight_providers or ["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"] if minimal_modules else None,
        )
        configure_insight_threads(self.app_insight, onnx_threads)
        self.app_insight.prepare(ctx_id=0, det_size=insight_det_size)
        self.latest_metrics: dict[str, float] = {
            "faces_ms": 0.0,
            "face_detect_ms": 0.0,
            "face_embed_ms": 0.0,
            "face_match_ms": 0.0,
            "face_embeddings": 0.0,
            "face_reused": 0.0,
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
        # Nome guardado pode ser de quem saiu do cadastro.
        self.reset_identities()
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
        # Partes do faces_ms. So o prefiltro separa deteccao e embedding: no
        # FaceAnalysis.get os dois saem juntos e ficam so no total.
        detect_ms = embed_ms = match_ms = 0.0
        if self.prefilter:
            # Mesmos argumentos e keypoints usados por FaceAnalysis.get.
            bboxes, keypoints = self.app_insight.det_model.detect(
                frame_context.processing_frame, max_num=0, metric="default"
            )
            faces = [Face(bbox=box[:4], det_score=box[4],
                          kps=keypoints[index] if keypoints is not None else None)
                     for index, box in enumerate(bboxes)]
            detect_ms = (time.perf_counter() - started_at) * 1000.0
            embeddings = 0
        else:
            faces = self.app_insight.get(frame_context.processing_frame)
            embeddings = len(faces)
        results: list[dict[str, Any]] = []
        ignored_faces: list[dict[str, Any]] = []
        accepted = []

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
            accepted.append((face, bbox_processing, bbox_original))

        # Rosto no mesmo lugar de um reconhecido ha menos de reuse_seconds herda
        # nome e semelhanca, sem embedding. So o prefiltro separa as duas etapas.
        now = getattr(frame_context, "observed_at", None) or time.monotonic()
        generation = self._identity_generation
        reuse = self.prefilter and self.reuse_seconds > 0
        reused = self._assign_identities(
            [bbox for _, bbox, _ in accepted], now) if reuse else {}
        identities = []

        for index, (face, bbox_processing, bbox_original) in enumerate(accepted):
            if index in reused:
                entry = reused[index]
                name, best_score = entry["name"], entry["score"]
                recognized_at = entry["recognized_at"]
            else:
                if self.prefilter:
                    embed_started_at = time.perf_counter()
                    for task, model in self.app_insight.models.items():
                        if task != "detection":
                            model.get(frame_context.processing_frame, face)
                    embed_ms += (time.perf_counter() - embed_started_at) * 1000.0
                    embeddings += 1
                match_started_at = time.perf_counter()
                name, best_score = self._match_face(face.normed_embedding)
                match_ms += (time.perf_counter() - match_started_at) * 1000.0
                recognized_at = now
            identities.append({"bbox": bbox_processing, "name": name,
                               "score": best_score, "recognized_at": recognized_at})

            results.append(
                {
                    "name": name,
                    "bbox": bbox_original,
                    "confidence": float(best_score),
                    "det_confidence": float(getattr(face, "det_score", 0.0)),
                }
            )

        # Cadastro trocado no meio do frame: nao guardar nomes da base antiga.
        if reuse and generation == self._identity_generation:
            self._identities = identities
        if self.debug_pipeline:
            self.latest_metrics["ignored_faces"] = float(len(ignored_faces))
        self.latest_ignored_faces = ignored_faces
        self.latest_metrics.update({
            "faces_ms": (time.perf_counter() - started_at) * 1000.0,
            "face_detect_ms": detect_ms,
            "face_embed_ms": embed_ms,
            "face_match_ms": match_ms,
            "face_embeddings": float(embeddings),
            "face_reused": float(len(reused)),
        })
        return results

    def reset_identities(self) -> None:
        """Esquece as identidades guardadas: conexao nova ou cadastro alterado."""
        self._identities = []
        self._identity_generation += 1

    def _assign_identities(
        self, boxes: list[list[int]], now: float
    ) -> dict[int, dict[str, Any]]:
        """Liga cada rosto a no maximo uma identidade valida, pela maior sobreposicao."""
        valid = [entry for entry in self._identities
                 if now - entry["recognized_at"] < self.reuse_seconds]
        pairs = sorted(
            ((self._iou(box, entry["bbox"]), index, slot)
             for index, box in enumerate(boxes)
             for slot, entry in enumerate(valid)),
            reverse=True,
        )
        assigned: dict[int, dict[str, Any]] = {}
        used: set[int] = set()
        for overlap, index, slot in pairs:
            if overlap < self.REUSE_MIN_IOU:
                break
            if index not in assigned and slot not in used:
                assigned[index] = valid[slot]
                used.add(slot)
        return assigned

    @staticmethod
    def _iou(first: list[int], second: list[int]) -> float:
        width = min(first[2], second[2]) - max(first[0], second[0])
        height = min(first[3], second[3]) - max(first[1], second[1])
        if width <= 0 or height <= 0:
            return 0.0
        intersection = width * height
        area = ((first[2] - first[0]) * (first[3] - first[1])
                + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
        return intersection / area if area > 0 else 0.0

    def detect_persons(
        self,
        frame_context: FrameContext,
    ) -> list[dict[str, Any]]:
        import time

        started_at = time.perf_counter()
        if self.model_yolo is None:
            if not os.path.exists(self.yolo_model_path):
                raise FileNotFoundError(f"Modelo YOLO nao encontrado: {self.yolo_model_path}")
            self.model_yolo = YOLO(self.yolo_model_path)
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
