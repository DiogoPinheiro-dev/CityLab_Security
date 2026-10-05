import base64
import logging
import math
import time
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from App.settings import (
    COOLDOWN_ALERTA_GESTO_SECONDS,
    COOLDOWN_ALUNO_SECONDS,
    COOLDOWN_NAO_ALUNO_SECONDS,
)

logger = logging.getLogger(__name__)


class EventLogger:
    # Rosto desconhecido nao tem nome: continua o episodio quem aparece ate esta
    # distancia, em pixels, de um desconhecido ja gravado no frame anterior. E a
    # mesma distancia do casamento de reserva do rastreador de gestos.
    unknown_face_match_distance = 120.0
    # Rosto esperando o reconhecimento em segundo plano: o mesmo texto de
    # FaceRecognitionService.PENDING_NAME. Nao gera evento, nem o rosto marcado
    # com "pending", de nome ainda nao confirmado.
    pending_name = "VERIFICANDO"

    def __init__(self, logs_collection: Any) -> None:
        self.logs_collection = logs_collection
        self.cooldowns = {
            "ALUNO": COOLDOWN_ALUNO_SECONDS,
            "NAO_ALUNO": COOLDOWN_NAO_ALUNO_SECONDS,
            "ALERTA_GESTO": COOLDOWN_ALERTA_GESTO_SECONDS,
        }
        self.last_logged: dict[str, float] = {}
        self.in_flight: set[str] = set()
        # Alertas ja gravados no episodio atual, por track e alerta: enquanto
        # seguem no frame seguinte, nao geram outro evento.
        self.logged_gesture_episodes: set[tuple[int, str]] = set()
        # Rostos ja gravados no episodio atual, como nos gestos: alunos pelo
        # nome, desconhecidos pelo centro do rosto no ultimo frame.
        self.logged_students: set[str] = set()
        self.logged_unknown_centers: list[tuple[float, float]] = []

    async def log_face_events(
        self,
        frame: np.ndarray,
        faces: Iterable[dict[str, Any]],
    ) -> None:
        students, unknown, pending = [], [], 0
        for face in faces:
            bbox = face.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            name = str(face.get("name") or "NAO ALUNO")
            # Sem nome ainda, ou com o nome que veio de outra posicao e espera a
            # confirmacao neste rosto: aparece na tela, mas nao vira evento.
            if name == self.pending_name or face.get("pending"):
                pending += 1
                continue
            (unknown if name == "NAO ALUNO" else students).append(
                (name, bbox, face.get("confidence")))

        # Enquanto o aluno segue no frame seguinte, nao grava de novo; quando
        # some de um frame, a proxima aparicao grava outra vez. Um rosto ainda
        # sem nome pode ser quem ja estava: com ele, nenhum episodio fecha.
        if not pending:
            self.logged_students &= {name for name, *_ in students}
        for name, bbox, confidence in students:
            if name in self.logged_students or not self._should_log("ALUNO", name):
                continue
            payload = self._build_base_payload(
                name=name, event_type="ALUNO", bbox=bbox, confidence=confidence)
            # So entra no episodio depois de gravado: falha tenta de novo.
            if await self._insert_event("ALUNO", name, payload):
                self.logged_students.add(name)

        previous, self.logged_unknown_centers = self.logged_unknown_centers, []
        for name, bbox, confidence in unknown:
            center = self._bbox_center(bbox)
            match = next((point for point in previous
                          if math.dist(point, center) <= self.unknown_face_match_distance),
                         None)
            if match is not None:
                previous.remove(match)
                self.logged_unknown_centers.append(center)
                continue
            identity = self._bbox_identity("unknown", bbox)
            if not self._should_log("NAO_ALUNO", identity):
                continue
            payload = self._build_base_payload(
                name=name, event_type="NAO_ALUNO", bbox=bbox, confidence=confidence)
            payload["imagem_url"] = self._crop_to_base64(frame, bbox)
            if await self._insert_event("NAO_ALUNO", identity, payload):
                self.logged_unknown_centers.append(center)
        if pending:
            self.logged_unknown_centers.extend(previous)

    async def log_gesture_events(
        self,
        frame: np.ndarray,
        gestures: Iterable[dict[str, Any]],
    ) -> None:
        candidates = []
        for gesture in gestures:
            alerts = [str(alert) for alert in gesture.get("alerts", []) if str(alert)]
            bbox = gesture.get("bbox")
            if not alerts or not bbox or len(bbox) != 4:
                continue

            track_id = int(gesture.get("track_id", -1))
            candidates.append((track_id, alerts, bbox, gesture.get("confidence"),
                               gesture.get("alert_evidence") or {}))

        # O cooldown em segundos nao deduplica quando o frame dura mais que ele:
        # cada alerta so termina o episodio quando some do frame, e um alerta
        # que continua nao grava de novo quando outro entra ou sai.
        self.logged_gesture_episodes &= {
            (track_id, alert) for track_id, alerts, *_ in candidates for alert in alerts}
        for track_id, alerts, bbox, confidence, evidence in candidates:
            new_alerts = [alert for alert in alerts
                          if (track_id, alert) not in self.logged_gesture_episodes]
            if not new_alerts:
                continue
            identity = f"{track_id}:{'|'.join(sorted(new_alerts))}"
            if not self._should_log("ALERTA_GESTO", identity):
                continue

            payload = self._build_base_payload(
                name=f"TRACK_{track_id}",
                event_type="ALERTA_GESTO",
                bbox=bbox,
                confidence=confidence,
            )
            payload["alertas"] = alerts
            payload["alertas_novos"] = new_alerts
            # No Pi, 2 observacoes em 6 s nao valem o mesmo que 12 em 0,4 s.
            payload["evidencia"] = [{"alerta": alert, **evidence[alert]}
                                    for alert in alerts if alert in evidence]
            payload["track_id"] = track_id
            payload["imagem_url"] = self._crop_to_base64(frame, bbox)
            # So entra no episodio depois de gravado: falha tenta de novo.
            if await self._insert_event("ALERTA_GESTO", identity, payload):
                self.logged_gesture_episodes.update((track_id, alert) for alert in new_alerts)

    def reset_episodes(self) -> None:
        """Nova sessao de stream: o proximo alerta ou rosto abre um episodio novo."""
        self.logged_gesture_episodes.clear()
        self.logged_students.clear()
        self.logged_unknown_centers = []

    async def _insert_event(self, event_type: str, identity: str, payload: dict) -> bool:
        key = f"{event_type}:{identity}"
        # Check + reserva sem await: atomico no event loop do servidor.
        if key in self.in_flight or not self._should_log(event_type, identity):
            return False
        self.in_flight.add(key)
        try:
            await self.logs_collection.insert_one(payload)
            self.last_logged[key] = time.monotonic()
            return True
        except Exception:
            logger.exception("Falha ao gravar evento %s; stream preservado.", event_type)
            return False
        finally:
            # Inclui cancelamento: a proxima deteccao pode tentar novamente.
            self.in_flight.discard(key)

    def _should_log(self, event_type: str, identity: str) -> bool:
        cooldown = self.cooldowns.get(event_type, 0.0)
        now = time.monotonic()
        expired = [key for key, last in self.last_logged.items()
                   if now - last > self.cooldowns.get(key.split(":", 1)[0], 0.0)]
        for key in expired:
            del self.last_logged[key]
        key = f"{event_type}:{identity}"
        if key in self.in_flight:
            return False
        last = self.last_logged.get(key)
        if last is not None and (now - last) <= cooldown:
            return False

        return True

    def _build_base_payload(
        self,
        name: str,
        event_type: str,
        bbox: list[int],
        confidence: Any,
    ) -> dict[str, Any]:
        now = datetime.now()
        return {
            "nome": name,
            "tipo": event_type,
            "data_hora_formatada": now.strftime("%d/%m/%Y - %H:%M:%S"),
            "data_hora_raw": now,
            "confidence": float(confidence) if confidence is not None else None,
            "bbox": [int(value) for value in bbox],
        }

    def _crop_to_base64(self, frame: np.ndarray, bbox: list[int]) -> str | None:
        x1, y1, x2, y2 = [int(value) for value in bbox]
        frame_h, frame_w = frame.shape[:2]
        crop = frame[
            max(0, y1):min(frame_h, y2),
            max(0, x1):min(frame_w, x2),
        ]
        if crop.size == 0:
            return None

        ok, buffer = cv2.imencode(".jpg", crop)
        if not ok:
            return None

        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _bbox_center(self, bbox: list[int]) -> tuple[float, float]:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _bbox_identity(self, prefix: str, bbox: list[int]) -> str:
        x1, y1, x2, y2 = [int(value) for value in bbox]
        return f"{prefix}:{x1 // 40}:{y1 // 40}:{x2 // 40}:{y2 // 40}"
