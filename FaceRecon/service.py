import logging
import os
import pickle
import time
from typing import Any
from typing import Optional

import cv2
import insightface
import numpy as np
from ultralytics import YOLO  # type: ignore


def setup_logger(script_dir: str) -> tuple[logging.Logger, logging.Logger, str]:
    base_log_directory = os.path.join(script_dir, "historico")
    text_log_directory = os.path.join(base_log_directory, "escrito")
    image_log_directory = os.path.join(base_log_directory, "imagem-nao-aluno")

    os.makedirs(text_log_directory, exist_ok=True)
    os.makedirs(image_log_directory, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger_alunos = logging.getLogger("AlunosLogger")
    logger_alunos.setLevel(logging.INFO)
    if not logger_alunos.handlers:
        handler_alunos = logging.FileHandler(
            os.path.join(text_log_directory, "reconhecimento_alunos.log"),
            mode="a",
            encoding="utf-8",
        )
        handler_alunos.setFormatter(formatter)
        logger_alunos.addHandler(handler_alunos)

    logger_alertas = logging.getLogger("AlertasLogger")
    logger_alertas.setLevel(logging.WARNING)
    if not logger_alertas.handlers:
        handler_alertas = logging.FileHandler(
            os.path.join(text_log_directory, "alertas_nao_alunos.log"),
            mode="a",
            encoding="utf-8",
        )
        handler_alertas.setFormatter(formatter)
        logger_alertas.addHandler(handler_alertas)

    return logger_alunos, logger_alertas, image_log_directory


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 0.01)
    table = np.array(
        [((value / 255.0) ** inv_gamma) * 255 for value in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


class FaceRecognitionService:
    def __init__(
        self,
        base_dir: Optional[str] = None,
        database_path: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
        similarity_threshold: float = 0.52,
        scale_factor: float = 0.5,
        gamma_value: float = 1.2,
        log_cooldown_seconds: float = 1.0,
        enable_logging: bool = True,
        face_model_name: str = "buffalo_l",
        insight_providers: Optional[list[str]] = None,
        insight_det_size: tuple[int, int] = (320, 320),
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
        self.scale_factor = scale_factor
        self.gamma_value = gamma_value
        self.log_cooldown_seconds = log_cooldown_seconds
        self.enable_logging = enable_logging

        self.logger_alunos: Optional[logging.Logger] = None
        self.logger_alertas: Optional[logging.Logger] = None
        self.image_log_directory: Optional[str] = None
        if self.enable_logging:
            (
                self.logger_alunos,
                self.logger_alertas,
                self.image_log_directory,
            ) = setup_logger(self.base_dir)

        self.known_face_embeddings = np.empty((0, 512), dtype=np.float32)
        self.known_face_names: list[str] = []
        self._load_database()

        self.model_yolo = YOLO(self.yolo_model_path)
        self.app_insight = insightface.app.FaceAnalysis(
            name=face_model_name,
            providers=insight_providers or ["CPUExecutionProvider"],
        )
        self.app_insight.prepare(ctx_id=0, det_size=insight_det_size)
        self.recently_logged: dict[str, float] = {}

    def _load_database(self) -> None:
        if not os.path.exists(self.database_path):
            self._log_warning(
                f"Arquivo da base de dados nao encontrado: {self.database_path}"
            )
            return

        with open(self.database_path, "rb") as file:
            data = pickle.load(file)

        embeddings = data.get("embeddings", [])
        names = data.get("names", [])

        if embeddings:
            self.known_face_embeddings = np.asarray(embeddings, dtype=np.float32)
        self.known_face_names = list(names)

    def replace_known_faces(
        self, names: list[str], embeddings: list[np.ndarray] | np.ndarray
    ) -> None:
        self.known_face_names = list(names)
        if len(embeddings) == 0:
            self.known_face_embeddings = np.empty((0, 512), dtype=np.float32)
            return

        self.known_face_embeddings = np.asarray(embeddings, dtype=np.float32)

    def prepare_frame(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        adjusted_frame = adjust_gamma(frame, gamma=self.gamma_value)
        small_frame = cv2.resize(
            adjusted_frame,
            (0, 0),
            fx=self.scale_factor,
            fy=self.scale_factor,
        )
        return {
            "original_frame": frame,
            "adjusted_frame": adjusted_frame,
            "small_frame": small_frame,
        }

    def recognize_faces(
        self,
        frame: np.ndarray,
        prepared_frame: Optional[dict[str, np.ndarray]] = None,
        with_logging: bool = True,
    ) -> list[dict[str, Any]]:
        prepared = prepared_frame or self.prepare_frame(frame)
        faces = self.app_insight.get(prepared["small_frame"])
        results: list[dict[str, Any]] = []

        for face in faces:
            name, best_score = self._match_face(face.normed_embedding)
            bbox = face.bbox.astype(int)

            results.append(
                {
                    "name": name,
                    "bbox": [int(coord / self.scale_factor) for coord in bbox],
                    "confidence": float(best_score),
                }
            )

            if with_logging:
                self._register_detection(name, bbox, frame)

        return results

    def detect_persons(
        self,
        frame: np.ndarray,
        prepared_frame: Optional[dict[str, np.ndarray]] = None,
    ) -> list[dict[str, Any]]:
        prepared = prepared_frame or self.prepare_frame(frame)
        results_yolo = self.model_yolo(
            prepared["small_frame"],
            classes=[0],
            verbose=False,
        )

        persons: list[dict[str, Any]] = []
        for result in results_yolo:
            for box in result.boxes:
                bbox_person = box.xyxy[0].numpy().astype(int)
                persons.append(
                    {
                        "bbox": [
                            int(coord / self.scale_factor)
                            for coord in bbox_person
                        ],
                        "confidence": float(box.conf[0]),
                    }
                )
        return persons

    def process_frame(
        self,
        frame: np.ndarray,
        detect_faces: bool = True,
        detect_persons: bool = True,
        with_logging: bool = True,
    ) -> dict[str, Any]:
        prepared = self.prepare_frame(frame)
        response: dict[str, Any] = {"faces": [], "persons": []}

        if detect_faces:
            response["faces"] = self.recognize_faces(
                frame,
                prepared_frame=prepared,
                with_logging=with_logging,
            )

        if detect_persons:
            response["persons"] = self.detect_persons(
                frame,
                prepared_frame=prepared,
            )

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

    def _register_detection(
        self,
        name: str,
        bbox: np.ndarray,
        original_frame: np.ndarray,
    ) -> None:
        current_time = time.time()
        last_logged_at = self.recently_logged.get(name)
        if last_logged_at and (
            current_time - last_logged_at <= self.log_cooldown_seconds
        ):
            return

        self.recently_logged[name] = current_time

        if name == "NAO ALUNO":
            self._save_unknown_face_image(bbox, original_frame, current_time)
            return

        if self.logger_alunos:
            self.logger_alunos.info("RECONHECIDO: %s", name)

    def _save_unknown_face_image(
        self,
        bbox: np.ndarray,
        original_frame: np.ndarray,
        current_time: float,
    ) -> None:
        if not self.image_log_directory:
            return

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        timestamp_ms = f"{timestamp}_{int(current_time * 1000) % 1000}"
        img_name = f"ALERTA_NAO_ALUNO_{timestamp_ms}.jpg"
        save_path = os.path.join(self.image_log_directory, img_name)

        inverse_scale = 1 / self.scale_factor
        h_full, w_full = original_frame.shape[:2]
        orig_x1 = max(0, int(bbox[0] * inverse_scale))
        orig_y1 = max(0, int(bbox[1] * inverse_scale))
        orig_x2 = min(w_full, int(bbox[2] * inverse_scale))
        orig_y2 = min(h_full, int(bbox[3] * inverse_scale))
        cropped_face = original_frame[orig_y1:orig_y2, orig_x1:orig_x2].copy()

        if cropped_face.size > 0:
            cv2.imwrite(save_path, cropped_face)
            self._log_warning(
                f"ALERTA: Pessoa nao cadastrada. Imagem salva: {save_path}"
            )
        else:
            self._log_warning(
                "ALERTA: Pessoa nao cadastrada. Falha ao salvar (rosto pequeno)."
            )

    def _log_warning(self, message: str) -> None:
        if self.logger_alertas:
            self.logger_alertas.warning(message)


def create_face_service(**kwargs: Any) -> FaceRecognitionService:
    return FaceRecognitionService(**kwargs)
