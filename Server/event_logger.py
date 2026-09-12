import base64
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


class EventLogger:
    def __init__(self, logs_collection: Any) -> None:
        self.logs_collection = logs_collection
        self.cooldowns = {
            "ALUNO": COOLDOWN_ALUNO_SECONDS,
            "NAO_ALUNO": COOLDOWN_NAO_ALUNO_SECONDS,
            "ALERTA_GESTO": COOLDOWN_ALERTA_GESTO_SECONDS,
        }
        self.last_logged: dict[str, float] = {}

    async def log_face_events(
        self,
        frame: np.ndarray,
        faces: Iterable[dict[str, Any]],
    ) -> None:
        for face in faces:
            bbox = face.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            name = str(face.get("name") or "NAO ALUNO")
            event_type = "ALUNO" if name != "NAO ALUNO" else "NAO_ALUNO"
            identity = name if event_type == "ALUNO" else self._bbox_identity("unknown", bbox)
            if not self._should_log(event_type, identity):
                continue

            payload = self._build_base_payload(
                name=name,
                event_type=event_type,
                bbox=bbox,
                confidence=face.get("confidence"),
            )
            if event_type == "NAO_ALUNO":
                payload["imagem_url"] = self._crop_to_base64(frame, bbox)

            await self.logs_collection.insert_one(payload)

    async def log_gesture_events(
        self,
        frame: np.ndarray,
        gestures: Iterable[dict[str, Any]],
    ) -> None:
        for gesture in gestures:
            alerts = [str(alert) for alert in gesture.get("alerts", []) if str(alert)]
            bbox = gesture.get("bbox")
            if not alerts or not bbox or len(bbox) != 4:
                continue

            track_id = int(gesture.get("track_id", -1))
            identity = f"{track_id}:{'|'.join(sorted(alerts))}"
            if not self._should_log("ALERTA_GESTO", identity):
                continue

            payload = self._build_base_payload(
                name=f"TRACK_{track_id}",
                event_type="ALERTA_GESTO",
                bbox=bbox,
                confidence=gesture.get("confidence"),
            )
            payload["alertas"] = alerts
            payload["track_id"] = track_id
            payload["imagem_url"] = self._crop_to_base64(frame, bbox)
            await self.logs_collection.insert_one(payload)

    def _should_log(self, event_type: str, identity: str) -> bool:
        cooldown = self.cooldowns.get(event_type, 0.0)
        now = time.time()
        key = f"{event_type}:{identity}"
        last = self.last_logged.get(key)
        if last is not None and (now - last) <= cooldown:
            return False

        self.last_logged[key] = now
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

    def _bbox_identity(self, prefix: str, bbox: list[int]) -> str:
        x1, y1, x2, y2 = [int(value) for value in bbox]
        return f"{prefix}:{x1 // 40}:{y1 // 40}:{x2 // 40}:{y2 // 40}"
