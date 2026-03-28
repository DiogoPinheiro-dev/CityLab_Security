import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
POSE_MODEL_PATH = BASE_DIR / "yolov8n-pose.pt"

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from ultralytics import YOLO

try:
    from .detector import GestureAnalyzer
except ImportError:
    from detector import GestureAnalyzer


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


def main():
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo de pose nao encontrado em: {POSE_MODEL_PATH}")

    model = YOLO(str(POSE_MODEL_PATH))
    analyzer = GestureAnalyzer(fps=30)

    video_source = 1
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened() or not cap.read()[0]:
        print("Camera externa nao encontrada. Iniciando webcam nativa...")
        cap.release()
        video_source = 0
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Erro ao abrir qualquer fonte de video.")
        return

    print("Iniciando deteccao... Pressione 'q' para sair.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fim do video ou erro na leitura.")
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        current_tracks = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
                boxes = _to_numpy(result.boxes.xyxy)
                track_ids = _to_int_list(result.boxes.id)
                keypoints_batch = _to_numpy(result.keypoints.data)

                for box, track_id, keypoints in zip(boxes, track_ids, keypoints_batch):
                    current_tracks.append(track_id)
                    alerts = analyzer.analyze(track_id, keypoints, box)

                    x1, y1, x2, y2 = map(int, box)
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

                    if alerts:
                        for i, alert in enumerate(alerts):
                            text_y = max(0, y1 - 10) + (i * 20)
                            cv2.putText(
                                frame,
                                alert,
                                (x1, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )

        analyzer.clean_old_tracks(current_tracks)
        cv2.imshow("Suspicious Gesture Recognition (CityLab)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
