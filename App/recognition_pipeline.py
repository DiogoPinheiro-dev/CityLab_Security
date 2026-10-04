import logging
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

import numpy as np

from App.frame_context import build_frame_context
from App.inference_runtime import configure_torch_threads, configure_opencv_threads, ensure_torch_threads
from App.settings import (
    CITYLAB_ALLOW_PARTIAL_PIPELINE,
    CITYLAB_ENABLE_FACE_SERVICE,
    CITYLAB_ENABLE_GESTURE_SERVICE,
    DEBUG_PIPELINE,
    ENABLE_PERFORMANCE_METRICS,
    EXPERIMENTAL_GRAYSCALE,
    PIPELINE_MAX_WORKERS,
    PIPELINE_RUN_IN_PARALLEL,
    PIPELINE_SHARED_PERSON_POSE,
    PROCESS_SCALE,
    TORCH_NUM_THREADS,
    OPENCV_NUM_THREADS,
)

if TYPE_CHECKING:
    from App.FaceRecon.service import FaceRecognitionService
    from App.GestureRecon.service import GestureRecognitionService

logger = logging.getLogger(__name__)

# Importados sob demanda: InsightFace e Ultralytics so carregam se o servico for
# criado, o que permite subir em modo parcial sem essas dependencias.
def _load_face_service_class() -> type["FaceRecognitionService"]:
    from App.FaceRecon.service import FaceRecognitionService

    return FaceRecognitionService

def _load_gesture_service_class() -> type["GestureRecognitionService"]:
    from App.GestureRecon.service import GestureRecognitionService

    return GestureRecognitionService

class UnifiedRecognitionService:
    """
    Orquestra reconhecimento facial e de gestos sobre o mesmo frame.
    """

    # Total do rosto e suas partes, com as mesmas chaves em todo frame.
    FACE_METRICS = ("faces_ms", "face_detect_ms", "face_embed_ms", "face_match_ms",
                    "face_embeddings", "face_reused")
    GESTURE_METRICS = ("gestures_ms", "hands_ms", "pose_ms", "motion_ratio", "pose_skipped",
                       "ncnn_threads")

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
        shared_person_pose: bool = PIPELINE_SHARED_PERSON_POSE,
        torch_threads: int = TORCH_NUM_THREADS,
        opencv_threads: int = OPENCV_NUM_THREADS,
        allow_partial_pipeline: bool = CITYLAB_ALLOW_PARTIAL_PIPELINE,
    ) -> None:
        self.allow_partial_pipeline = allow_partial_pipeline
        self.shared_person_pose = shared_person_pose
        self.face_service = face_service or self._create_face_service()
        self.gesture_service = gesture_service or self._create_gesture_service()
        configure_torch_threads(torch_threads)
        configure_opencv_threads(opencv_threads)
        self.torch_threads = torch_threads
        self.run_in_parallel = run_in_parallel
        self.process_scale = process_scale
        self.experimental_grayscale = experimental_grayscale
        self.debug_pipeline = debug_pipeline
        self.enable_performance_metrics = enable_performance_metrics or debug_pipeline
        self.executor: Optional[ThreadPoolExecutor] = None
        self.max_workers = max(1, max_workers)
        self._last_frame_ended_at: Optional[float] = None

        if self.run_in_parallel:
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="pipeline",
                initializer=configure_torch_threads,
                initargs=(torch_threads,),
            )

    def _create_face_service(self) -> Optional["FaceRecognitionService"]:
        if not CITYLAB_ENABLE_FACE_SERVICE:
            logger.warning("Face service desativado por CITYLAB_ENABLE_FACE_SERVICE.")
            return None

        try:
            return _load_face_service_class()(lazy_person_model=self.shared_person_pose)
        except Exception as exc:
            if not self.allow_partial_pipeline:
                raise

            logger.warning("Face service indisponivel: %s", exc)
            return None

    def _create_gesture_service(self) -> Optional["GestureRecognitionService"]:
        if not CITYLAB_ENABLE_GESTURE_SERVICE:
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
        use_shared_pose = (self.shared_person_pose and detect_gestures
                           and self.gesture_service is not None)
        if detect_persons and self.face_service is not None and not use_shared_pose:
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
                use_shared_pose=use_shared_pose,
            )
        else:
            if detect_faces and self.face_service is not None:
                faces = self.face_service.recognize_faces(frame_context)
                self._copy_face_metrics(metrics, ran=True)
            else:
                self._copy_face_metrics(metrics, ran=False)

            if detect_gestures and self.gesture_service is not None:
                metrics["gesture_torch_threads"] = ensure_torch_threads(self.torch_threads)
                gestures = self.gesture_service.detect_gestures(
                    frame_context,
                    person_bboxes=None if use_shared_pose else persons,
                )
                self._copy_gesture_metrics(metrics, ran=True)
            else:
                self._copy_gesture_metrics(metrics, ran=False)

        # Mesmas chaves em todo caminho, para o coletor contar todos os frames.
        metrics.setdefault("face_worker", -1)
        metrics.setdefault("gesture_worker", -1)
        metrics.setdefault("gesture_torch_threads", 0)

        if use_shared_pose and detect_persons:
            persons = self.gesture_service.latest_persons

        if self.gesture_service is not None:
            # Rosto visto agora protege o proximo frame do gate de movimento:
            # quem esta parado na cena nao pode sumir por falta de movimento.
            self.gesture_service.note_external_presence(bool(faces))

        payload = self._merge_payloads(
            face_payload={"faces": faces, "persons": persons},
            gesture_payload={"gestures": gestures},
        )
        learned = getattr(self.face_service, "latest_learned", None) if detect_faces else None
        forgotten = getattr(self.face_service, "latest_forgotten", None) if detect_faces else None
        if learned or forgotten:
            # Referencias aprendidas e trocadas neste frame, para o servidor
            # gravar no banco; tem embedding, entao nao vao para o cliente.
            payload["aprendidos"] = {"novos": list(learned or []),
                                     "removidos": list(forgotten or [])}

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
                "shared_person_pose": use_shared_pose,
                "run_in_parallel": self.run_in_parallel,
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
        use_shared_pose: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        face_future = None
        gesture_future = None

        if detect_faces and self.face_service is not None and self.executor is not None:
            face_future = self.executor.submit(
                self._recognize_faces_on_worker,
                frame_context,
            )
        if (
            detect_gestures
            and self.gesture_service is not None
            and self.executor is not None
        ):
            gesture_future = self.executor.submit(
                self._detect_gestures_on_worker,
                frame_context,
                None if use_shared_pose else persons,
            )

        faces: list[dict[str, Any]] = []
        gestures: list[dict[str, Any]] = []

        # Mesmo com falha em um modelo, aguardar o outro antes de liberar estado.
        wait([future for future in (face_future, gesture_future) if future is not None])
        if face_future is not None and self.face_service is not None:
            faces, metrics["face_worker"] = face_future.result()
            self._copy_face_metrics(metrics, ran=True)
        else:
            self._copy_face_metrics(metrics, ran=False)

        if gesture_future is not None and self.gesture_service is not None:
            gestures, metrics["gesture_worker"], metrics["gesture_torch_threads"] = (
                gesture_future.result()
            )
            self._copy_gesture_metrics(metrics, ran=True)
        else:
            self._copy_gesture_metrics(metrics, ran=False)

        return faces, gestures

    @staticmethod
    def _worker_index() -> int:
        """Indice do worker do executor, pelo nome da thread; -1 fora dele."""
        prefix, _, index = threading.current_thread().name.rpartition("_")
        return int(index) if prefix == "pipeline" and index.isdigit() else -1

    def _copy_face_metrics(self, metrics: dict[str, float], ran: bool) -> None:
        for name in self.FACE_METRICS:
            metrics[name] = self.face_service.latest_metrics.get(name, 0.0) if ran else 0.0

    def _copy_gesture_metrics(self, metrics: dict[str, float], ran: bool) -> None:
        for name in self.GESTURE_METRICS:
            metrics[name] = self.gesture_service.latest_metrics.get(name, 0.0) if ran else 0.0

    def _recognize_faces_on_worker(self, frame_context: Any) -> tuple[list[dict[str, Any]], int]:
        return self.face_service.recognize_faces(frame_context), self._worker_index()

    def _detect_gestures_on_worker(
        self,
        frame_context: Any,
        person_bboxes: Optional[list[dict[str, Any]]],
    ) -> tuple[list[dict[str, Any]], int, int]:
        # O worker que rodou o primeiro track() herdou o limite do Ultralytics;
        # reaplicar o configurado deixa a pose com a mesma velocidade nos dois.
        threads = ensure_torch_threads(self.torch_threads)
        gestures = self.gesture_service.detect_gestures(frame_context, person_bboxes)
        return gestures, self._worker_index(), threads

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

    def reset_face_identities(self) -> None:
        reset = getattr(self.face_service, "reset_identities", None)
        if reset is not None:
            reset()

    def reset_gesture_history(self) -> None:
        if self.gesture_service is not None:
            self.gesture_service.analyzer.clean_old_tracks([])
            self.gesture_service.last_track_centers.clear()
