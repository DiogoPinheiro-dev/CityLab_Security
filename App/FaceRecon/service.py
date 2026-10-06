import os
import pickle
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from typing import Optional

import insightface
from insightface.app.common import Face
import numpy as np
from ultralytics import YOLO  # type: ignore

from App.frame_context import FrameContext
from App.settings import (DEBUG_PIPELINE, FACE_ASYNC_RECOGNITION, FACE_DETECTOR_PATH,
                          FACE_EMBED_FULL_FRAME, FACE_LEARN_FROM_STREAM, FACE_LEARN_INTERVAL_SECONDS,
                          FACE_LEARNED_PER_PERSON, FACE_LEARNED_RETENTION_DAYS,
                          FACE_MIN_CONFIDENCE, FACE_MIN_HEIGHT, FACE_MIN_WIDTH,
                          FACE_MINIMAL_MODULES, FACE_PREFILTER, FACE_REUSE_SECONDS,
                          ONNX_ALLOW_SPINNING, ONNX_INTRA_OP_THREADS, PROJECT_ROOT)
from App.inference_runtime import configure_insight_threads


class FaceRecognitionService:
    # Sobreposicao minima entre caixas de rosto de frames seguidos para tratar
    # como o mesmo rosto.
    REUSE_MIN_IOU = 0.5
    # "NAO ALUNO" com semelhanca daqui ate o limite costuma ser aluno de lado, e
    # nao estranho: nao herda o nome, e o rosto e reconhecido de novo no frame
    # seguinte. Rostos de outras pessoas nao passaram de 0,16, e o de um aluno
    # virado ficou de 0,37 a 0,48 (03 e 04/10/2026). No lugar do aluno que
    # estava ali, fica com o nome dele sem confirmar (_keep_turned_students).
    UNKNOWN_RECHECK_MIN_SIMILARITY = 0.30
    # Rosto esperando o reconhecimento em segundo plano: aparece assim, sem
    # evento. O mesmo texto esta em Server/event_logger.py e Client/stream.js.
    PENDING_NAME = "VERIFICANDO"
    # Com o reconhecimento em segundo plano, o rosto que andou mais que a
    # sobreposicao leva o nome mais proximo ate esta distancia em pixels do
    # frame original, a mesma do registro de eventos para seguir um
    # desconhecido (Server/event_logger.py), e e reconhecido de novo. Sem
    # outro rosto ou outra identidade por perto, o nome segue confirmado.
    CARRY_MAX_DISTANCE_PX = 120.0
    # Semelhanca minima de cada foto do cadastro com a media das outras. Em 10
    # fotos da mesma pessoa a pior ficou em 0,50 (03/10/2026); abaixo disso a
    # foto e tratada como de outra pessoa, para nao misturar dois rostos.
    REGISTRATION_MIN_SIMILARITY = 0.3
    # Semelhanca minima para um rosto do stream virar referencia da pessoa. Na
    # simulacao de 03/10/2026 aprendeu com 9 de 10 fotos de cadastro, e 7 rostos
    # de outras pessoas nao passaram de 0,16 com as referencias aprendidas.
    LEARN_MIN_SIMILARITY = 0.60
    # Daqui para cima, o rosto do stream e copia de uma referencia ja guardada
    # da pessoa. Frames da mesma cena do video de carga ficam em 0,78 na
    # mediana: so os quase iguais passam (simulacao de 03/10/2026).
    LEARN_COPY_SIMILARITY = 0.90

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
        onnx_allow_spinning: bool = ONNX_ALLOW_SPINNING,
        embed_full_frame: bool = FACE_EMBED_FULL_FRAME,
        learn_from_stream: bool = FACE_LEARN_FROM_STREAM,
        learned_per_person: int = FACE_LEARNED_PER_PERSON,
        learn_interval_seconds: float = FACE_LEARN_INTERVAL_SECONDS,
        learned_retention_days: int = FACE_LEARNED_RETENTION_DAYS,
        face_detector_path: str = FACE_DETECTOR_PATH,
        async_recognition: bool = FACE_ASYNC_RECOGNITION,
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
        # Embedding no frame original em vez da imagem reduzida; so com prefiltro.
        self.embed_full_frame = embed_full_frame
        # Reconhecimento em segundo plano: so com o prefiltro, que separa a
        # deteccao do embedding. Um por vez, numa thread propria.
        self.async_recognition = async_recognition and prefilter
        self._recognition_pool = None
        if self.async_recognition:
            from concurrent.futures import ThreadPoolExecutor
            self._recognition_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rosto")
        self._job: Optional[dict[str, Any]] = None
        self._job_count = 0
        # Zero desliga: todo rosto aceito gera embedding em todo frame.
        self.reuse_seconds = reuse_seconds
        self._identities: list[dict[str, Any]] = []
        self._identity_generation = 0

        self.known_face_embeddings = np.empty((0, 512), dtype=np.float32)
        self.known_face_names: list[str] = []
        # Referencias aprendidas no stream, por nome. Quem grava no banco e o
        # servidor, pelas novas de cada frame em latest_learned.
        self.learn_from_stream = learn_from_stream
        self.learned_per_person = learned_per_person
        self.learn_interval = timedelta(seconds=learn_interval_seconds)
        self.learned_retention = (timedelta(days=learned_retention_days)
                                  if learned_retention_days > 0 else None)
        self.learned: dict[str, list[dict[str, Any]]] = {}
        self._learned_embeddings = np.empty((0, 512), dtype=np.float32)
        self._learned_names: list[str] = []
        self.latest_learned: list[dict[str, Any]] = []
        self.latest_forgotten: list[str] = []
        self._load_database()

        self.model_yolo = None if lazy_person_model else YOLO(self.yolo_model_path)
        self.app_insight = insightface.app.FaceAnalysis(
            name=face_model_name,
            providers=insight_providers or ["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"] if minimal_modules else None,
        )
        if face_detector_path:
            self._replace_detector(face_detector_path, insight_providers)
        configure_insight_threads(self.app_insight, onnx_threads,
                                  allow_spinning=onnx_allow_spinning)
        self.app_insight.prepare(ctx_id=0, det_size=insight_det_size)
        self.latest_metrics: dict[str, float] = {
            "faces_ms": 0.0,
            "face_detect_ms": 0.0,
            "face_embed_ms": 0.0,
            "face_match_ms": 0.0,
            "face_embeddings": 0.0,
            "face_reused": 0.0,
            "face_pending": 0.0,
            "persons_ms": 0.0,
        }
        self.latest_ignored_faces: list[dict[str, Any]] = []

    def _replace_detector(self, path: str, providers: Optional[list[str]]) -> None:
        """Troca so o detector do pacote; o reconhecedor, e o cadastro, ficam.

        Antes das threads do ONNX e do prepare, que entao valem para ele tambem.
        """
        detector_path = Path(path).expanduser()
        if not detector_path.is_absolute():
            detector_path = PROJECT_ROOT / detector_path
        if not detector_path.is_file():
            raise FileNotFoundError(f"Detector de rosto nao encontrado: {detector_path}")
        detector = insightface.model_zoo.get_model(
            str(detector_path), providers=providers or ["CPUExecutionProvider"])
        if getattr(detector, "taskname", None) != "detection":
            raise ValueError(f"{detector_path.name} nao e um detector de rosto do InsightFace.")
        self.app_insight.models["detection"] = detector
        self.app_insight.det_model = detector

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
        # Referencias aprendidas so de quem continua no cadastro.
        known = set(self.known_face_names)
        self.learned = {name: refs for name, refs in self.learned.items() if name in known}
        self._rebuild_learned()
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
        self.latest_learned, self.latest_forgotten = [], []
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
        boxes = [bbox for _, bbox, _ in accepted]
        if self.async_recognition:
            done = self._collect_recognition()
            if done is not None:
                embed_ms += done["embed_ms"]
                match_ms += done["match_ms"]
                embeddings += 1 if "name" in done else 0
                self._apply_recognition(done, now)
            # Toda identidade conta, inclusive a que espera o reconhecimento.
            paired = self._pair_identities(boxes, self._identities)
            carried, unsure = self._carry_identities(accepted, paired)
            reused = {}
        else:
            reused = self._assign_identities(boxes, now) if reuse else {}
        identities = []
        # "NAO ALUNO" quase reconhecidos neste frame, que podem ser aluno de perfil.
        near_misses = []

        for index, (face, bbox_processing, bbox_original) in enumerate(accepted):
            job, confirmed = None, True
            previous_job, job_carried, seen_at = None, False, None
            if self.async_recognition:
                entry = paired.get(index, carried.get(index))
                previous_job = entry.get("job") if entry is not None else None
                if index in unsure and previous_job is not None:
                    # Com outro rosto ou outra identidade perto, o resultado fica
                    # com o rosto mais proximo, mas so confirma na posicao atual.
                    entry["job_carried"] = True
                name, best_score, recognized_at, job, inherited, confirmed = self._recognize_async(
                    entry, index in carried, index in unsure, face, frame_context, now)
                job_carried = bool(
                    entry is not None and previous_job is not None and job == previous_job
                    and entry.get("job_carried"))
                if inherited:
                    reused[index] = entry
                # Sem confirmar, o rosto guarda quando o nome foi visto confirmado.
                if entry is not None:
                    seen_at = entry.get("seen_at")
            elif index in reused:
                entry = reused[index]
                name, best_score = entry["name"], entry["score"]
                recognized_at = entry["recognized_at"]
            else:
                target = face
                if self.prefilter:
                    embed_started_at = time.perf_counter()
                    image = frame_context.processing_frame
                    if self.embed_full_frame:
                        image = frame_context.original_frame
                        target = self._face_on_original(face, frame_context)
                    for task, model in self.app_insight.models.items():
                        if task != "detection":
                            model.get(image, target)
                    embed_ms += (time.perf_counter() - embed_started_at) * 1000.0
                    embeddings += 1
                match_started_at = time.perf_counter()
                name, best_score = self._match_face(target.normed_embedding)
                match_ms += (time.perf_counter() - match_started_at) * 1000.0
                recognized_at = now
                # So de embedding novo: nome herdado pelo reuso nao ensina nada.
                if self.learn_from_stream:
                    self._maybe_learn(name, best_score, target.normed_embedding)
                if (reuse and name == "NAO ALUNO"
                        and best_score >= self.UNKNOWN_RECHECK_MIN_SIMILARITY):
                    near_misses.append((index, target.normed_embedding, best_score,
                                        bbox_original))
            identities.append({"bbox": bbox_processing, "center": self._center(bbox_original),
                               "name": name, "score": best_score,
                               "recognized_at": recognized_at, "job": job,
                               "confirmed": confirmed, "job_carried": job_carried,
                               "seen_at": now if confirmed else seen_at})

            results.append(
                {
                    "name": name,
                    "bbox": bbox_original,
                    "confidence": float(best_score),
                    "det_confidence": float(getattr(face, "det_score", 0.0)),
                    # Nome ainda nao confirmado neste rosto: aparece, sem evento.
                    "pending": not confirmed,
                }
            )

        if near_misses:
            self._keep_turned_students(near_misses, results, identities, now)
        # Cadastro trocado no meio do frame: nao guardar nomes da base antiga.
        if (reuse or self.async_recognition) and generation == self._identity_generation:
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
            "face_pending": float(sum(face["pending"] for face in results)),
        })
        return results

    def _keep_turned_students(self, near_misses: list[tuple[int, Any, float, list[int]]],
                              results: list[dict[str, Any]],
                              identities: list[dict[str, Any]], now: float) -> None:
        """Aluno que virou o rosto continua ele, sem confirmar, no lugar onde estava.

        De perfil, o aluno sai "NAO ALUNO" com semelhanca de 0,30 ate o limite, e
        o registro de eventos gravava um NAO_ALUNO e reabria o episodio dele. Se a
        pessoa mais parecida com o rosto e o aluno que estava ate
        CARRY_MAX_DISTANCE_PX dele no frame anterior, visto confirmado ha menos de
        reuse_seconds, o rosto fica com o nome do aluno sem confirmar: aparece
        como "verificando", nao vira evento e nao fecha o episodio. O frame
        seguinte reconhece de novo; o prazo conta da ultima vez em que o aluno foi
        visto confirmado, pelo reconhecimento ou pelo reuso.
        """
        import math

        present = {result["name"] for result in results if not result["pending"]}
        students = [entry for entry in self._identities
                    if entry["name"] not in ("NAO ALUNO", self.PENDING_NAME)
                    and entry["name"] not in present and "center" in entry
                    and now - self._seen_at(entry) < self.reuse_seconds]
        pairs = sorted(
            (math.dist(self._center(bbox_original), entry["center"]), position, slot)
            for position, (_, _, _, bbox_original) in enumerate(near_misses)
            for slot, entry in enumerate(students))
        used_faces: set[int] = set()
        used_students: set[int] = set()
        for distance, position, slot in pairs:
            if distance > self.CARRY_MAX_DISTANCE_PX:
                break
            if position in used_faces or slot in used_students:
                continue
            index, embedding, score, _ = near_misses[position]
            student = students[slot]
            # So quando o aluno e o mais parecido, mesmo abaixo do limite.
            if self._person_score(student["name"], embedding) < score - 1e-6:
                continue
            used_faces.add(position)
            used_students.add(slot)
            results[index].update(name=student["name"], pending=True)
            identities[index].update(name=student["name"], confirmed=False,
                                     recognized_at=student["recognized_at"],
                                     seen_at=self._seen_at(student))

    @staticmethod
    def _seen_at(entry: dict[str, Any]) -> float:
        """Ultima vez em que o nome foi visto confirmado neste rosto."""
        seen_at = entry.get("seen_at")
        return entry["recognized_at"] if seen_at is None else seen_at

    def _recognize_async(self, entry: Optional[dict[str, Any]], carried: bool, unsure: bool,
                         face: Any, frame_context: FrameContext, now: float
                         ) -> tuple[str, float, float, Optional[int], bool, bool]:
        """Nome do rosto sem esperar o embedding.

        Devolve nome, semelhanca, hora do reconhecimento, job, se herdou como no
        reuso e se o nome esta confirmado neste rosto. Herda como no reuso; o que
        espera reconhecimento segue esperando; o resto pede um, se a thread
        estiver livre. Enquanto isso fica o nome de antes, ou VERIFICANDO para
        quem acabou de aparecer. O nome que veio de outra posicao (carried) nao
        e herdado sem reconhecer de novo; ele so continua confirmado se nao ha
        duvida de que e o mesmo rosto (ver _still_confirmed).
        """
        if entry is not None and entry.get("job") is not None:
            return (entry["name"], entry["score"], entry["recognized_at"], entry["job"],
                    False, self._still_confirmed(entry, carried, unsure, now))
        if entry is not None and not carried and self._reusable(entry, now):
            return (entry["name"], entry["score"], entry["recognized_at"], entry.get("job"),
                    entry.get("job") is None, entry.get("confirmed", True))
        job = self._submit_recognition(face, frame_context, now) if self._job is None else None
        if entry is not None:
            return (entry["name"], entry["score"], entry["recognized_at"], job, False,
                    self._still_confirmed(entry, carried, unsure, now))
        return self.PENDING_NAME, 0.0, now, job, False, False

    def _still_confirmed(self, entry: dict[str, Any], carried: bool, unsure: bool,
                         now: float) -> bool:
        """O nome confirmado antes vale neste rosto enquanto ele e reconhecido de novo.

        No mesmo lugar, sim. Levado pela distancia, so se ele era o unico rosto
        perto do nome e o nome, a unica identidade perto dele (unsure falso), e
        se o ultimo reconhecimento tem menos de reuse_seconds. No Pi, em
        05/10/2026, exigir a sobreposicao deixou quem anda em "verificando" em
        27 de 30 frames: o resultado chega quando a pessoa ja saiu do lugar.
        """
        if not entry.get("confirmed", True) or unsure:
            return False
        return not carried or now - entry["recognized_at"] < self.reuse_seconds

    def _carry_identities(self, accepted: list[Any], paired: dict[int, dict[str, Any]]
                          ) -> tuple[dict[int, dict[str, Any]], set[int]]:
        """Rosto que andou mais que a sobreposicao: o nome mais proximo, ate o limite.

        So entre os rostos sem par e as identidades com nome que sobraram, do
        par mais proximo para o mais longe, um para um. Devolve tambem os rostos
        levados com duvida: com outro rosto sem par perto da mesma identidade,
        ou outra identidade livre perto do mesmo rosto.
        """
        import math

        taken = {id(entry) for entry in paired.values()}
        free = [entry for entry in self._identities
                if id(entry) not in taken
                and (entry["name"] != self.PENDING_NAME or entry.get("job") is not None)
                and "center" in entry]
        pairs = sorted(
            (math.dist(self._center(bbox_original), entry["center"]), index, slot)
            for index, (_, _, bbox_original) in enumerate(accepted) if index not in paired
            for slot, entry in enumerate(free))
        carried: dict[int, dict[str, Any]] = {}
        used: set[int] = set()
        for distance, index, slot in pairs:
            if distance > self.CARRY_MAX_DISTANCE_PX:
                break
            if index not in carried and slot not in used:
                carried[index] = free[slot]
                used.add(slot)
        near = [(index, slot) for distance, index, slot in pairs
                if distance <= self.CARRY_MAX_DISTANCE_PX]
        unsure = {index for index, entry in carried.items()
                  if sum(1 for face, _ in near if face == index) > 1
                  or sum(1 for _, slot in near if free[slot] is entry) > 1}
        return carried, unsure

    @staticmethod
    def _center(bbox: list[int]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _submit_recognition(self, face: Any, frame_context: FrameContext, now: float) -> int:
        """Manda o embedding do rosto para a thread do reconhecimento; devolve o job."""
        if self.embed_full_frame:
            image, target = frame_context.original_frame, self._face_on_original(face, frame_context)
        else:
            image = frame_context.processing_frame
            target = Face(bbox=face.bbox, kps=face.kps, det_score=face.det_score)
        self._job_count += 1
        self._job = {"id": self._job_count, "generation": self._identity_generation,
                     "observed_at": now,
                     "future": self._recognition_pool.submit(self._embed, image, target)}
        return self._job_count

    def _embed(self, image: Any, target: Any) -> tuple[Any, float]:
        """Roda na thread do reconhecimento: so o embedding, sem estado compartilhado."""
        import time

        started_at = time.perf_counter()
        for task, model in self.app_insight.models.items():
            if task != "detection":
                model.get(image, target)
        return target.normed_embedding, (time.perf_counter() - started_at) * 1000.0

    def _collect_recognition(self) -> Optional[dict[str, Any]]:
        """Reconhecimento em segundo plano que terminou: nome, semelhanca e aprendizado."""
        import time

        job = self._job
        if job is None or not job["future"].done():
            return None
        self._job = None
        try:
            embedding, embed_ms = job["future"].result()
        except Exception as exc:
            print(f"[AVISO] Reconhecimento do rosto em segundo plano falhou: {exc}")
            return {"id": job["id"], "embed_ms": 0.0, "match_ms": 0.0}
        if job["generation"] != self._identity_generation:
            # Conexao nova ou cadastro trocado: o nome seria da base anterior.
            return None
        started_at = time.perf_counter()
        name, score = self._match_face(embedding)
        # So de embedding novo: nome herdado pelo reuso nao ensina nada.
        if self.learn_from_stream:
            self._maybe_learn(name, score, embedding)
        return {"id": job["id"], "name": name, "score": score, "embedding": embedding,
                "recognized_at": job["observed_at"], "embed_ms": embed_ms,
                "match_ms": (time.perf_counter() - started_at) * 1000.0}

    def _apply_recognition(self, done: dict[str, Any], now: float) -> None:
        """Poe o resultado no rosto que esperava por ele, se ainda esta na cena."""
        for entry in self._identities:
            if entry.get("job") == done["id"]:
                entry["job"] = None
                if "name" not in done:
                    continue
                if self._turned_away(entry, done, now):
                    # Aluno de perfil, como em _keep_turned_students: segue com o
                    # nome dele, sem confirmar, e o rosto e reconhecido de novo.
                    entry.update(score=done["score"], confirmed=False,
                                 seen_at=self._seen_at(entry))
                    continue
                # Resultado levado com duvida, com outro rosto ou outra
                # identidade perto, nao confirma a pessoa na nova posicao;
                # a proxima inferencia faz essa confirmacao.
                entry.update(name=done["name"], score=done["score"],
                             recognized_at=done["recognized_at"],
                             confirmed=not entry.get("job_carried", False))

    def _turned_away(self, entry: dict[str, Any], done: dict[str, Any], now: float) -> bool:
        """O resultado e "NAO ALUNO" quase reconhecido como o aluno deste rosto."""
        return (done["name"] == "NAO ALUNO"
                and done["score"] >= self.UNKNOWN_RECHECK_MIN_SIMILARITY
                and entry["name"] not in ("NAO ALUNO", self.PENDING_NAME)
                and now - self._seen_at(entry) < self.reuse_seconds
                and self._person_score(entry["name"], done["embedding"]) >= done["score"] - 1e-6)

    def close(self) -> None:
        """Encerra a thread do reconhecimento em segundo plano."""
        pool = getattr(self, "_recognition_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            self._recognition_pool = None

    @classmethod
    def average_embedding(cls, embeddings: list[Any]) -> tuple[Optional[np.ndarray], Optional[int]]:
        """Media normalizada das fotos do cadastro, ou o indice da primeira que destoa."""
        vectors = np.asarray(embeddings, dtype=np.float32)
        if len(vectors) > 1:
            for index in range(len(vectors)):
                others = np.delete(vectors, index, axis=0).mean(axis=0)
                others /= np.linalg.norm(others)
                if float(vectors[index] @ others) < cls.REGISTRATION_MIN_SIMILARITY:
                    return None, index
        mean = vectors.mean(axis=0)
        return mean / np.linalg.norm(mean), None

    @staticmethod
    def _face_on_original(face: Any, frame_context: FrameContext) -> Any:
        """O mesmo rosto nas coordenadas do frame original, para o embedding."""
        sx, sy = frame_context.scale_x, frame_context.scale_y
        x1, y1, x2, y2 = face.bbox[:4]
        return Face(bbox=np.asarray([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32),
                    kps=np.asarray([[x * sx, y * sy] for x, y in face.kps], dtype=np.float32),
                    det_score=face.det_score)

    def reset_identities(self) -> None:
        """Esquece as identidades guardadas: conexao nova ou cadastro alterado."""
        self._identities = []
        self._identity_generation += 1

    def _assign_identities(
        self, boxes: list[list[int]], now: float
    ) -> dict[int, dict[str, Any]]:
        """Liga cada rosto a no maximo uma identidade valida, pela maior sobreposicao."""
        return self._pair_identities(
            boxes, [entry for entry in self._identities if self._reusable(entry, now)])

    def _reusable(self, entry: dict[str, Any], now: float) -> bool:
        """Identidade que um rosto no mesmo lugar herda, sem reconhecer de novo."""
        return (now - entry["recognized_at"] < self.reuse_seconds
                and entry.get("job") is None and entry["name"] != self.PENDING_NAME
                and entry.get("confirmed", True)
                and not (entry["name"] == "NAO ALUNO"
                         and entry["score"] >= self.UNKNOWN_RECHECK_MIN_SIMILARITY))

    def _pair_identities(self, boxes: list[list[int]],
                         entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        """Cada rosto com no maximo uma das identidades, pela maior sobreposicao."""
        pairs = sorted(
            ((self._iou(box, entry["bbox"]), index, slot)
             for index, box in enumerate(boxes)
             for slot, entry in enumerate(entries)),
            reverse=True,
        )
        assigned: dict[int, dict[str, Any]] = {}
        used: set[int] = set()
        for overlap, index, slot in pairs:
            if overlap < self.REUSE_MIN_IOU:
                break
            if index not in assigned and slot not in used:
                assigned[index] = entries[slot]
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
        # Vale a mais parecida entre o cadastro e as referencias aprendidas.
        best_name, best_score = None, 0.0
        for embeddings, names in ((self.known_face_embeddings, self.known_face_names),
                                  (self._learned_embeddings, self._learned_names)):
            if len(embeddings) == 0:
                continue
            scores = np.dot(embeddings, live_embedding)
            index = int(np.argmax(scores))
            if best_name is None or float(scores[index]) > best_score:
                best_name, best_score = names[index], float(scores[index])

        if best_name is None:
            return "NAO ALUNO", 0.0
        if best_score > self.similarity_threshold:
            return best_name, best_score

        return "NAO ALUNO", best_score

    def _cadastro_score(self, name: str, live_embedding: np.ndarray) -> float:
        """Semelhanca com o cadastro da pessoa, sem as referencias aprendidas."""
        return max((float(np.dot(embedding, live_embedding))
                    for embedding, known in zip(self.known_face_embeddings, self.known_face_names)
                    if known == name), default=0.0)

    def _person_score(self, name: str, live_embedding: np.ndarray) -> float:
        """Semelhanca com a pessoa: o cadastro e as referencias aprendidas dela."""
        learned = max((float(np.dot(embedding, live_embedding))
                       for embedding, known in zip(self._learned_embeddings, self._learned_names)
                       if known == name), default=0.0)
        return max(self._cadastro_score(name, live_embedding), learned)

    def _maybe_learn(self, name: str, score: float, embedding: np.ndarray) -> None:
        """Guarda o rosto como referencia da pessoa, se a folga e o cadastro permitem.

        Com as vagas cheias, a nova troca a referencia menos parecida com o
        cadastro, se for mais parecida que ela: uma referencia errada e a
        primeira a sair. Medir pela nota do aprendizado deixava as referencias
        parecidas demais entre si (simulacao de 03/10/2026). A copia de uma
        referencia guardada so disputa a vaga dela: no Pi, o mesmo video passado
        de novo fazia a copia expulsar uma referencia diferente (03/10/2026).
        """
        now = datetime.now()
        self._prune_learned(now)
        if name == "NAO ALUNO" or score < self.LEARN_MIN_SIMILARITY:
            return
        # Perto do cadastro tambem: as referencias nao derivam para outra pessoa.
        anchor = self._cadastro_score(name, embedding)
        if anchor <= self.similarity_threshold:
            return
        refs = self.learned.get(name, [])
        if refs and now - max(ref["aprendido_em"] for ref in refs) < self.learn_interval:
            return
        rival = self._copy_of(refs, embedding)
        if rival is None and len(refs) >= self.learned_per_person:
            rival = min(refs, key=lambda ref: self._cadastro_score(name, ref["embedding"]))
        if rival is not None:
            if anchor <= self._cadastro_score(name, rival["embedding"]):
                return
            refs = [ref for ref in refs if ref is not rival]
            self.latest_forgotten.append(rival["id"])
        entry = {"id": uuid.uuid4().hex, "nome": name, "semelhanca": float(score),
                 "semelhanca_cadastro": anchor, "aprendido_em": now,
                 "embedding": np.asarray(embedding, dtype=np.float32)}
        self.learned[name] = [*refs, entry]
        self.latest_learned.append(entry)
        self._rebuild_learned()

    def _copy_of(self, refs: list[dict[str, Any]],
                 embedding: np.ndarray) -> Optional[dict[str, Any]]:
        """A referencia mais parecida com o rosto, se ele for copia dela."""
        if not refs:
            return None
        scores = [float(np.dot(ref["embedding"], embedding)) for ref in refs]
        index = int(np.argmax(scores))
        return refs[index] if scores[index] >= self.LEARN_COPY_SIMILARITY else None

    def _prune_learned(self, now: datetime) -> None:
        """Esquece na memoria o que o banco apaga pelo prazo."""
        if self.learned_retention is None or not self.learned:
            return
        limit = now - self.learned_retention
        kept = {name: [ref for ref in refs if ref["aprendido_em"] > limit]
                for name, refs in self.learned.items()}
        if any(len(kept[name]) != len(refs) for name, refs in self.learned.items()):
            self.learned = {name: refs for name, refs in kept.items() if refs}
            self._rebuild_learned()

    def _rebuild_learned(self) -> None:
        entries = [ref for refs in self.learned.values() for ref in refs]
        self._learned_names = [ref["nome"] for ref in entries]
        self._learned_embeddings = (
            np.asarray([ref["embedding"] for ref in entries], dtype=np.float32)
            if entries else np.empty((0, 512), dtype=np.float32))

    def replace_learned(self, entries: list[dict[str, Any]]) -> list[str]:
        """Carrega as referencias guardadas no banco; devolve os ids que ficaram de fora.

        Fica de fora a de quem saiu do cadastro, a que hoje nao entraria, longe
        do cadastro, e a copia de outra mais parecida com ele. Por pessoa ficam
        as mais parecidas com o cadastro, ate o limite.
        """
        known = set(self.known_face_names)
        expired = (datetime.now() - self.learned_retention
                   if self.learned_retention is not None else None)
        learned: dict[str, list[dict[str, Any]]] = {}
        discarded: list[str] = []
        for entry in entries:
            # Vencida nao ocupa vaga: o proprio banco a apaga pelo prazo.
            if expired is not None and entry["aprendido_em"] <= expired:
                continue
            embedding = np.asarray(entry["embedding"], dtype=np.float32)
            if (entry["nome"] not in known
                    or self._cadastro_score(entry["nome"], embedding) <= self.similarity_threshold):
                discarded.append(entry["id"])
                continue
            learned.setdefault(entry["nome"], []).append({**entry, "embedding": embedding})
        for name, refs in learned.items():
            refs.sort(key=lambda ref: self._cadastro_score(name, ref["embedding"]), reverse=True)
            kept: list[dict[str, Any]] = []
            for ref in refs:
                if (len(kept) >= self.learned_per_person
                        or self._copy_of(kept, ref["embedding"]) is not None):
                    discarded.append(ref["id"])
                else:
                    kept.append(ref)
            learned[name] = kept
        self.learned = learned
        self._rebuild_learned()
        return discarded

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
