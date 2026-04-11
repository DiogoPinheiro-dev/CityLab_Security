import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
POSE_MODEL_PATH = BASE_DIR / "GestureRecon" / "yolov8n-pose.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from App.FaceRecon.reconhecimento import ProcessadorCV
from App.GestureRecon.detector import GestureAnalyzer


def _to_numpy(data: Any):
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        return data.numpy()
    return np.asarray(data)


def _to_int_list(data: Any):
    if hasattr(data, "int"):
        data = data.int()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "tolist"):
        return [int(value) for value in data.tolist()]
    return [int(value) for value in data]


class ReconhecimentoUnificado:
    def __init__(self):
        self.face_processor = ProcessadorCV()
        self.gesture_analyzer = GestureAnalyzer(fps=30)

        if not POSE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo de pose nao encontrado em: {POSE_MODEL_PATH}. "
                "Adicione o arquivo yolov8n-pose.pt em App/GestureRecon."
            )
        self.pose_model = YOLO(str(POSE_MODEL_PATH))

        self.last_face_results = {"faces": [], "persons": []}
        self.last_gesture_results = []
        self.frame_count = 0

        # Reaproveita os ultimos resultados em alguns frames para aliviar carga.
        self.face_skip_frames = 2
        self.gesture_skip_frames = 1

    def processar_gestos(self, frame):
        results = self.pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        current_tracks = []
        gesture_results = []

        if results and len(results) > 0:
            result = results[0]
            if (
                result.boxes is not None
                and result.boxes.id is not None
                and result.keypoints is not None
            ):
                boxes = _to_numpy(result.boxes.xyxy)
                track_ids = _to_int_list(result.boxes.id)
                keypoints_batch = _to_numpy(result.keypoints.data)

                for box, track_id, keypoints in zip(boxes, track_ids, keypoints_batch):
                    current_tracks.append(track_id)
                    alerts = self.gesture_analyzer.analyze(track_id, keypoints, box)
                    gesture_results.append(
                        {
                            "track_id": track_id,
                            "bbox": [int(value) for value in box],
                            "alerts": alerts,
                        }
                    )

        self.gesture_analyzer.clean_old_tracks(current_tracks)
        return gesture_results

    def processar_frame(self, frame):
        should_run_faces = self.frame_count % (self.face_skip_frames + 1) == 0
        should_run_gestures = self.frame_count % (self.gesture_skip_frames + 1) == 0

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_faces = None
            future_gestures = None

            if should_run_faces:
                future_faces = executor.submit(self.face_processor.processar_frame, frame)

            if should_run_gestures:
                future_gestures = executor.submit(self.processar_gestos, frame)

            if future_faces is not None:
                self.last_face_results = future_faces.result()

            if future_gestures is not None:
                self.last_gesture_results = future_gestures.result()

        self.frame_count += 1

        return {
            "faces": self.last_face_results.get("faces", []),
            "persons": self.last_face_results.get("persons", []),
            "gestures": self.last_gesture_results,
        }


def desenhar_faces(frame, faces):
    for face in faces:
        name = face["name"]
        x1, y1, x2, y2 = face["bbox"]
        conf = face["confidence"]

        if name == "NAO ALUNO":
            color = (0, 0, 255)
            label = "NAO ALUNO"
        else:
            color = (0, 255, 0)
            label = f"{name} ({int(conf * 100)}%)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def desenhar_pessoas(frame, persons):
    for person in persons:
        px1, py1, px2, py2 = person["bbox"]
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 1)


def desenhar_gestos(frame, gestures):
    for person in gestures:
        x1, y1, x2, y2 = person["bbox"]
        track_id = person["track_id"]
        alerts = person["alerts"]

        color = (0, 0, 255) if alerts else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, max(0, y1 - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        for index, alert in enumerate(alerts):
            text_y = max(0, y1 - 10) + (index * 20)
            cv2.putText(
                frame,
                alert,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )


def main():
    recognizer = ReconhecimentoUnificado()

    print("[INFO] Abrindo camera")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel acessar a camera.")

    print("--- SISTEMA UNIFICADO INICIADO ---")
    print("Pressione 'q' na janela de video para sair.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERRO] Falha ao ler frame da camera")
                break

            results = recognizer.processar_frame(frame)

            desenhar_faces(frame, results["faces"])
            desenhar_pessoas(frame, results["persons"])
            desenhar_gestos(frame, results["gestures"])

            cv2.imshow("Reconhecimento Unificado", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupcao manual detectada")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Sistema encerrado corretamente.")


if __name__ == "__main__":
    main()
