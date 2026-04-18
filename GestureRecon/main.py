import os
import sys

import cv2
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT_DIR, "Ultralytics"))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from GestureRecon.detector import GestureAnalyzer
    from GestureRecon.hand_detector import HandDetector
except ImportError:
    from detector import GestureAnalyzer
    from hand_detector import HandDetector


def _keypoint_xy_conf(keypoints, index):
    keypoint = keypoints[index]
    return float(keypoint[0]), float(keypoint[1]), float(keypoint[2])


def _point_in_box(point, box):
    if box is None:
        return False

    point_x, point_y = point
    x1, y1, x2, y2 = box
    return x1 <= point_x <= x2 and y1 <= point_y <= y2


def _build_torso_box(keypoints):
    torso_indices = [5, 6, 11, 12]
    visible_points = []
    for index in torso_indices:
        x, y, conf = _keypoint_xy_conf(keypoints, index)
        if conf > 0.4:
            visible_points.append((x, y))

    if len(visible_points) < 2:
        return None

    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _match_hand_side(hand, left_wrist, right_wrist):
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


def associar_maos(box, keypoints, hand_detections):
    x1, y1, x2, y2 = [float(value) for value in box]
    body_width = max(1.0, x2 - x1)
    body_height = max(1.0, y2 - y1)
    expanded_x1 = x1 - (body_width * 0.15)
    expanded_y1 = y1 - (body_height * 0.1)
    expanded_x2 = x2 + (body_width * 0.15)
    expanded_y2 = y2 + (body_height * 0.1)

    torso_box = _build_torso_box(keypoints)
    left_wrist = _keypoint_xy_conf(keypoints, 9)
    right_wrist = _keypoint_xy_conf(keypoints, 10)

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

        side, distance = _match_hand_side(hand, left_wrist, right_wrist)
        if side is None:
            continue

        hand_data = {
            "bbox": hand["bbox"],
            "center": hand["center"],
            "closed": hand["closed"],
            "in_torso": _point_in_box(hand["center"], torso_box),
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

def main():
    # Inicializa o modelo de Pose. 
    # O 'yolov8n-pose.pt' é a versão "nano" (mais rápida, ideal para tempo real).
    # Baixará automaticamente se não existir.
    model = YOLO("yolov8n-pose.pt")
    
    # Inicializa o analisador de gestos
    analyzer = GestureAnalyzer(fps=30)
    hand_detector = HandDetector()
    
    # Tenta usar a câmera externa primeiro (índice 1, ou maior),
    # Se falhar ou não existir, usa a webcam nativa (índice 0).
    video_source = 1
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened() or not cap.read()[0]:
        print("Câmera externa não encontrada. Iniciando webcam nativa...")
        cap.release()
        video_source = 0
        cap = cv2.VideoCapture(video_source)
        
    if not cap.isOpened():
        print(f"Erro ao abrir qualquer fonte de vídeo.")
        return

    print("Iniciando detecção... Pressione 'q' para sair.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro na leitura.")
            break
            
        hand_detections = hand_detector.detect(frame)

        # Roda o YOLO Tracking no frame.
        # persist=True avisa o modelo que os frames pertencem ao mesmo vídeo.
        # tracker="bytetrack.yaml" usa o ByteTrack, excelente para multidões.
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        current_tracks = []

        if results and len(results) > 0:
            result = results[0]
            
            # Se encontrou pessoas e possui tracking IDs
            if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                keypoints_batch = result.keypoints.data.cpu().numpy() # [N_pessoas, 17, 3]

                for box, track_id, keypoints in zip(boxes, track_ids, keypoints_batch):
                    current_tracks.append(track_id)
                    
                    # Analisa o comportamento da pessoa com base nas suas coordenadas dos membros e bounding box
                    hand_context = associar_maos(box, keypoints, hand_detections)
                    alerts = analyzer.analyze(
                        track_id,
                        keypoints,
                        box,
                        hand_context=hand_context,
                    )
                    
                    # --- Desenho Visual ---
                    x1, y1, x2, y2 = map(int, box)
                    
                    severe_alert = any(
                        alert in {"Mao Fechada + Braco Estendido", "Mao Oculta"}
                        for alert in alerts
                    )
                    if severe_alert:
                        color = (0, 0, 255)
                    elif alerts:
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)
                        
                    # Desenha a Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Desenha o ID da pessoa
                    label = f"ID: {track_id}"
                    cv2.putText(frame, label, (x1, max(0, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Desenha os alertas um embaixo do outro
                    if alerts:
                        for i, alert in enumerate(alerts):
                            text_y = max(0, y1 - 10) + (i * 20)
                            cv2.putText(frame, alert, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    for matched_hand in hand_context.get("matched_hands", []):
                        hx1, hy1, hx2, hy2 = matched_hand["bbox"]
                        hand_color = (0, 0, 255) if matched_hand.get("closed") else (255, 255, 0)
                        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), hand_color, 1)
                            
        # Limpa da memória IDs antigos que não aparecem mais na tela
        analyzer.clean_old_tracks(current_tracks)
        
        # Mostra o frame
        cv2.imshow("Suspicious Gesture Recognition (CityLab)", frame)
        
        # Saída com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
