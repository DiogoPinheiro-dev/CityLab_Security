import os
from concurrent.futures import ThreadPoolExecutor

import cv2

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ultralytics"),
)

from ultralytics import YOLO

from FaceRecon.reconhecimento import ProcessadorCV
from GestureRecon.detector import GestureAnalyzer
from GestureRecon.hand_detector import HandDetector
from camera_auto_config import AutoImageOptimizer, configure_camera_capture


class ReconhecimentoUnificado:
    def __init__(self):
        self.face_processor = ProcessadorCV()
        self.gesture_analyzer = GestureAnalyzer(fps=30)
        self.hand_detector = HandDetector()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        pose_model_path = os.path.join(base_dir, "GestureRecon", "yolov8n-pose.pt")
        self.pose_model = YOLO(pose_model_path)
        self.image_optimizer = AutoImageOptimizer()

        self.last_face_results = {"faces": [], "persons": []}
        self.last_gesture_results = []
        self.frame_count = 0

        self.face_skip_frames = 2
        self.gesture_skip_frames = 0

    def processar_gestos(self, frame):
        processed_frame = self.image_optimizer.optimize(frame)
        hand_detections = self.hand_detector.detect(processed_frame)
        results = self.pose_model.track(
            processed_frame,
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
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                keypoints_batch = result.keypoints.data.cpu().numpy()

                for box, track_id, keypoints in zip(boxes, track_ids, keypoints_batch):
                    current_tracks.append(track_id)
                    hand_context = self.associar_maos(box, keypoints, hand_detections)
                    alerts = self.gesture_analyzer.analyze(
                        track_id,
                        keypoints,
                        box,
                        hand_context=hand_context,
                    )
                    gesture_results.append(
                        {
                            "track_id": track_id,
                            "bbox": [int(value) for value in box],
                            "alerts": alerts,
                            "hand_context": hand_context,
                        }
                    )

        self.gesture_analyzer.clean_old_tracks(current_tracks)
        return gesture_results

    def associar_maos(self, box, keypoints, hand_detections):
        x1, y1, x2, y2 = [float(value) for value in box]
        body_width = max(1.0, x2 - x1)
        body_height = max(1.0, y2 - y1)
        expanded_x1 = x1 - (body_width * 0.15)
        expanded_y1 = y1 - (body_height * 0.1)
        expanded_x2 = x2 + (body_width * 0.15)
        expanded_y2 = y2 + (body_height * 0.1)

        torso_box = self._build_torso_box(keypoints)
        left_wrist = self._keypoint_xy_conf(keypoints, 9)
        right_wrist = self._keypoint_xy_conf(keypoints, 10)

        context = {
            "left_visible": False,
            "right_visible": False,
            "left_closed": False,
            "right_closed": False,
            "left_in_torso": False,
            "right_in_torso": False,
            "matched_hands": [],
        }

        left_candidate = None
        right_candidate = None

        for hand in hand_detections:
            center_x, center_y = hand["center"]
            if not (
                expanded_x1 <= center_x <= expanded_x2
                and expanded_y1 <= center_y <= expanded_y2
            ):
                continue

            side, distance = self._match_hand_side(hand, left_wrist, right_wrist)
            if side is None:
                continue

            hand_data = {
                "bbox": hand["bbox"],
                "center": hand["center"],
                "closed": hand["closed"],
                "in_torso": self._point_in_box(hand["center"], torso_box),
            }

            if side == "left":
                if left_candidate is None or distance < left_candidate["distance"]:
                    hand_data["distance"] = distance
                    left_candidate = hand_data
            else:
                if right_candidate is None or distance < right_candidate["distance"]:
                    hand_data["distance"] = distance
                    right_candidate = hand_data

        if left_candidate is not None:
            context["left_visible"] = True
            context["left_closed"] = left_candidate["closed"]
            context["left_in_torso"] = left_candidate["in_torso"]
            context["matched_hands"].append(
                {
                    "side": "left",
                    "bbox": left_candidate["bbox"],
                    "closed": left_candidate["closed"],
                }
            )

        if right_candidate is not None:
            context["right_visible"] = True
            context["right_closed"] = right_candidate["closed"]
            context["right_in_torso"] = right_candidate["in_torso"]
            context["matched_hands"].append(
                {
                    "side": "right",
                    "bbox": right_candidate["bbox"],
                    "closed": right_candidate["closed"],
                }
            )

        return context

    def _build_torso_box(self, keypoints):
        torso_indices = [5, 6, 11, 12]
        visible_points = []
        for index in torso_indices:
            x, y, conf = self._keypoint_xy_conf(keypoints, index)
            if conf > 0.4:
                visible_points.append((x, y))

        if len(visible_points) < 2:
            return None

        xs = [point[0] for point in visible_points]
        ys = [point[1] for point in visible_points]
        return [min(xs), min(ys), max(xs), max(ys)]

    def _match_hand_side(self, hand, left_wrist, right_wrist):
        best_side = None
        best_distance = None

        for side, wrist in (("left", left_wrist), ("right", right_wrist)):
            wrist_x, wrist_y, wrist_conf = wrist
            if wrist_conf <= 0.35:
                continue

            hand_x, hand_y = hand["center"]
            distance = ((hand_x - wrist_x) ** 2 + (hand_y - wrist_y) ** 2) ** 0.5
            if best_distance is None or distance < best_distance:
                best_side = side
                best_distance = distance

        return best_side, best_distance

    def _keypoint_xy_conf(self, keypoints, index):
        keypoint = keypoints[index]
        return float(keypoint[0]), float(keypoint[1]), float(keypoint[2])

    def _point_in_box(self, point, box):
        if box is None:
            return False

        point_x, point_y = point
        x1, y1, x2, y2 = box
        return x1 <= point_x <= x2 and y1 <= point_y <= y2

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
        hand_context = person.get("hand_context", {})

        severe_alert = any(
            alert in {"Mao Fechada + Braco Estendido", "Mao Oculta"}
            for alert in alerts
        )
        color = (0, 0, 255) if severe_alert else ((0, 165, 255) if alerts else (0, 255, 0))
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
                color,
                2,
            )

        for matched_hand in hand_context.get("matched_hands", []):
            hx1, hy1, hx2, hy2 = matched_hand["bbox"]
            hand_color = (0, 0, 255) if matched_hand.get("closed") else (255, 255, 0)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), hand_color, 1)


def main():
    recognizer = ReconhecimentoUnificado()

    print("[INFO] Abrindo camera")
    cap = cv2.VideoCapture(0)
    selected_resolution = configure_camera_capture(cap)

    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel acessar a camera.")

    if selected_resolution is not None:
        print(
            f"[INFO] Camera configurada automaticamente em "
            f"{selected_resolution[0]}x{selected_resolution[1]}"
        )

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
