"""Grava um gesto real para o banco de replay, sem imagem.

Roda a pose e as maos do projeto na webcam local, ou num video gravado, e
guarda por frame o que o GestureAnalyzer recebe: keypoints, caixa e o estado
das maos. Em "maos" guarda tambem, por mao associada a pessoa, o tamanho no
quadro entregue ao detector, o gesto e a confianca do classificador e qual
condicao a marcou como fechada, alem dos indices normalizados da regra lateral.
Com --maos-frame-cheio, so as maos usam o quadro original. Com
--maos-recorte-punho, cada mao usa um recorte do quadro original ao redor do
punho visto pela pose. Nos dois casos, a pose continua na escala configurada.
Nao grava imagem, rosto nem nome. Uma pessoa por vez na frente da camera; com
mais de uma, fica a maior caixa. Feche antes a pagina do stream, que usa a
webcam.

Uso:
    python tools/record_gesture_sequence.py rendicao --saida resultados/gestos/rendicao.json
    python tools/record_gesture_sequence.py rendicao --video rendicao.mp4 --inicio 2 \\
        --saida resultados/gestos/rendicao.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gesture_replay import GESTURES

HAND_KEYS = ("left_visible", "right_visible", "left_closed", "right_closed",
             "left_in_torso", "right_in_torso")


def _webcam_frames(cv2, index, seconds):
    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        raise SystemExit("Webcam indisponivel: feche a pagina do stream no navegador.")
    started = time.monotonic()
    try:
        while (elapsed := time.monotonic() - started) < seconds:
            ok, image = camera.read()
            if not ok:
                raise SystemExit("Falha ao ler a webcam.")
            yield elapsed, image
    finally:
        camera.release()


def _video_frames(cv2, path, start, seconds):
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise SystemExit(f"Nao foi possivel abrir o video {path}")
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    # Cerca de 10 frames por segundo, como a gravacao pela webcam.
    step = max(1, round(fps / 10))
    index = 0
    try:
        while True:
            ok, image = video.read()
            if not ok:
                return
            moment = index / fps
            index += 1
            if moment < start or (index - 1) % step:
                continue
            if moment - start >= seconds:
                return
            # Lado maior em 640 px, como o frame que o cliente envia ao stream.
            scale = 640 / max(image.shape[:2])
            if scale < 1:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            yield moment - start, image
    finally:
        video.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gesto", choices=sorted([*GESTURES, "neutro"]))
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--segundos", type=float, default=20.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, help="video gravado no lugar da webcam")
    parser.add_argument("--inicio", type=float, default=0.0,
                        help="segundos do video a pular, enquanto a pessoa se posiciona")
    hand_mode = parser.add_mutually_exclusive_group()
    hand_mode.add_argument("--maos-frame-cheio", action="store_true",
                           help="roda so o detector de maos no frame original")
    hand_mode.add_argument("--maos-recorte-punho", action="store_true",
                           help="roda o detector em um recorte por punho")
    parser.add_argument("--fator-recorte-punho", type=float, default=0.85,
                        help="meio lado do recorte em multiplos do antebraco")
    args = parser.parse_args()
    if args.fator_recorte_punho <= 0:
        parser.error("--fator-recorte-punho deve ser maior que zero")

    import cv2
    from App.GestureRecon.service import GestureRecognitionService
    from App.frame_context import build_frame_context
    from App.settings import EXPERIMENTAL_GRAYSCALE, PROCESS_SCALE

    # Sem o gate: a gravacao precisa da pose em todo frame.
    service = GestureRecognitionService(motion_gate=False)
    # O rastreador do Ultralytics carrega a DLL do pacote lap, que a politica de
    # Controle de Aplicativo do Windows pode bloquear. Com uma pessoa por vez
    # basta a deteccao: as caixas vem direto do detector, e o servico numera o
    # track pelo centro da caixa.
    predict = service.pose_model.predict
    pose_results = []

    def predict_without_tracker(frame, persist=True, tracker=None, **options):
        del persist, tracker
        pose_results.clear()
        pose_results.extend(predict(frame, **options))
        return pose_results

    service.pose_model.track = predict_without_tracker
    observed = []
    analyze = service.analyzer.analyze
    associate = service._associate_hands
    detect_hands = service._detect_hands_in_body_roi
    detector = service.hand_detector
    hands = []
    frame_context = None

    def side_rule_metrics(hand):
        points = hand.get("landmarks", [])
        if len(points) < 21:
            return None

        distance = detector._distance
        wrist = points[0]
        index_mcp = points[5]
        middle_mcp = points[9]
        pinky_mcp = points[17]
        fingertips = [points[index] for index in (8, 12, 16, 20)]
        finger_mcps = [points[index] for index in (5, 9, 13, 17)]
        palm_center = (
            (index_mcp[0] + middle_mcp[0] + pinky_mcp[0]) / 3.0,
            (index_mcp[1] + middle_mcp[1] + pinky_mcp[1]) / 3.0,
        )
        palm_size = max(1.0, (
            distance(wrist, index_mcp)
            + distance(wrist, pinky_mcp)
            + distance(index_mcp, pinky_mcp)
        ) / 3.0)
        tip_to_center = [distance(tip, palm_center) / palm_size for tip in fingertips]
        tip_to_mcp = [distance(tip, mcp) / palm_size
                      for tip, mcp in zip(fingertips, finger_mcps)]
        compact_tips = sum(
            center < 1.15 and base < 1.2
            for center, base in zip(tip_to_center, tip_to_mcp)
        )
        spread = max(
            distance(fingertips[i], fingertips[j])
            for i in range(len(fingertips))
            for j in range(i + 1, len(fingertips))
        ) / palm_size
        average = sum(tip_to_center) / len(tip_to_center)
        thumb_to_palm = distance(points[4], palm_center) / palm_size
        thumb_to_mcp = distance(points[4], points[2]) / palm_size
        thumb_compact = thumb_to_palm < 1.35 or thumb_to_mcp < 0.95
        return {
            "palma_px": round(palm_size, 3),
            "pontas_centro": [round(value, 4) for value in tip_to_center],
            "pontas_base": [round(value, 4) for value in tip_to_mcp],
            "pontas_compactas": compact_tips,
            "media_pontas": round(average, 4),
            "abertura_pontas": round(spread, 4),
            "polegar_palmas": round(thumb_to_palm, 4),
            "polegar_base": round(thumb_to_mcp, 4),
            "polegar_compacto": thumb_compact,
            "ativa": (compact_tips >= 3 and average < 1.05
                      and spread < 1.35 and thumb_compact),
        }

    def closed_by(hand):
        points = hand.get("landmarks", [])
        return [name for name, hit in (
            ("classificador", hand.get("gesture_name") == "Closed_Fist"),
            ("regra_frente", detector._is_closed_fist(points)),
            ("regra_lado", detector._is_side_closed_fist(points))) if hit]

    def detect_hands_full_frame(_processing_frame, box):
        if frame_context is None:
            return []
        original_box = frame_context.clip_original_bbox(
            frame_context.map_bbox_to_original([int(value) for value in box])
        )
        detected = detect_hands(frame_context.original_frame, original_box)
        remapped = []
        for hand in detected:
            x1, y1, x2, y2 = hand["bbox"]
            remapped.append({
                **hand,
                "bbox": [
                    int(round(x1 / frame_context.scale_x)),
                    int(round(y1 / frame_context.scale_y)),
                    int(round(x2 / frame_context.scale_x)),
                    int(round(y2 / frame_context.scale_y)),
                ],
                "center": [
                    int(round(hand["center"][0] / frame_context.scale_x)),
                    int(round(hand["center"][1] / frame_context.scale_y)),
                ],
                "landmarks": [
                    (int(round(point[0] / frame_context.scale_x)),
                     int(round(point[1] / frame_context.scale_y)))
                    for point in hand.get("landmarks", [])
                ],
                "_tamanho_entrada_px": max(x2 - x1, y2 - y1),
                "_fechada_por": closed_by(hand),
                "_metricas_regra_lado": side_rule_metrics(hand),
            })
        return remapped

    def keypoints_for_box(box):
        if not pose_results:
            return None
        result = pose_results[0]
        if result.boxes is None or result.keypoints is None:
            return None
        boxes = service._to_numpy(result.boxes.xyxy)
        keypoints = service._to_numpy(result.keypoints.data)
        if not len(boxes) or not len(keypoints):
            return None
        target = [float(value) for value in box]
        index = min(
            range(min(len(boxes), len(keypoints))),
            key=lambda candidate: sum(
                abs(float(boxes[candidate][axis]) - target[axis])
                for axis in range(4)
            ),
        )
        return keypoints[index]

    def detect_hands_wrist_crops(_processing_frame, box):
        if frame_context is None:
            return []
        keypoints = keypoints_for_box(box)
        if keypoints is None:
            return []

        original_box = frame_context.clip_original_bbox(
            frame_context.map_bbox_to_original([int(value) for value in box])
        )
        body_width = max(1, original_box[2] - original_box[0])
        body_height = max(1, original_box[3] - original_box[1])
        frame = frame_context.original_frame
        frame_h, frame_w = frame.shape[:2]
        remapped = []

        for side, wrist_index, elbow_index in (
            ("left", 9, 7),
            ("right", 10, 8),
        ):
            wrist = keypoints[wrist_index]
            if float(wrist[2]) <= 0.35:
                continue
            wrist_x, wrist_y = frame_context.map_point_to_original(
                [float(wrist[0]), float(wrist[1])]
            )
            elbow = keypoints[elbow_index]
            if float(elbow[2]) > 0.35:
                elbow_x, elbow_y = frame_context.map_point_to_original(
                    [float(elbow[0]), float(elbow[1])]
                )
                forearm = ((wrist_x - elbow_x) ** 2 + (wrist_y - elbow_y) ** 2) ** 0.5
            else:
                forearm = max(body_width * 0.25, body_height * 0.16)

            half_side = max(32.0, forearm * args.fator_recorte_punho)
            half_side = min(half_side, max(body_width, body_height) * 0.35)
            x1 = max(0, int(round(wrist_x - half_side)))
            y1 = max(0, int(round(wrist_y - half_side)))
            x2 = min(frame_w, int(round(wrist_x + half_side)))
            y2 = min(frame_h, int(round(wrist_y + half_side)))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            for hand in detector.detect(crop):
                hand_x1, hand_y1, hand_x2, hand_y2 = hand["bbox"]
                original_hand = {
                    **hand,
                    "bbox": [
                        hand_x1 + x1,
                        hand_y1 + y1,
                        hand_x2 + x1,
                        hand_y2 + y1,
                    ],
                    "center": [hand["center"][0] + x1, hand["center"][1] + y1],
                    "landmarks": [
                        (point[0] + x1, point[1] + y1)
                        for point in hand.get("landmarks", [])
                    ],
                }
                remapped.append({
                    **original_hand,
                    "bbox": [
                        int(round(value / scale))
                        for value, scale in zip(
                            original_hand["bbox"],
                            (frame_context.scale_x, frame_context.scale_y,
                             frame_context.scale_x, frame_context.scale_y),
                        )
                    ],
                    "center": [
                        int(round(original_hand["center"][0] / frame_context.scale_x)),
                        int(round(original_hand["center"][1] / frame_context.scale_y)),
                    ],
                    "landmarks": [
                        (int(round(point[0] / frame_context.scale_x)),
                         int(round(point[1] / frame_context.scale_y)))
                        for point in original_hand.get("landmarks", [])
                    ],
                    "_tamanho_entrada_px": max(
                        hand_x2 - hand_x1,
                        hand_y2 - hand_y1,
                    ),
                    "_fechada_por": closed_by(original_hand),
                    "_metricas_regra_lado": side_rule_metrics(original_hand),
                    "_recorte_lado": side,
                    "_recorte_px": [x2 - x1, y2 - y1],
                })
        return remapped

    if args.maos_frame_cheio:
        service._detect_hands_in_body_roi = detect_hands_full_frame
    elif args.maos_recorte_punho:
        service._detect_hands_in_body_roi = detect_hands_wrist_crops

    def capture_hands(box, keypoints, hand_detections):
        # Chamado para cada pessoa logo antes do analisador: guarda o que o
        # detector viu nas maos que o servico associou a ela.
        context = associate(box, keypoints, hand_detections)
        hands.clear()
        for matched in context["matched_hands"]:
            hand = next((item for item in hand_detections if item["bbox"] == matched["bbox"]), None)
            if hand is None:
                continue
            x1, y1, x2, y2 = hand["bbox"]
            reasons = hand.get("_fechada_por")
            if reasons is None:
                reasons = closed_by(hand)
            metrics = hand.get("_metricas_regra_lado")
            if metrics is None:
                metrics = side_rule_metrics(hand)
            hands.append({
                "lado": matched["side"],
                "tamanho_px": hand.get("_tamanho_entrada_px", max(x2 - x1, y2 - y1)),
                "gesto": hand.get("gesture_name"),
                "confianca": round(float(hand.get("gesture_score") or 0.0), 3),
                "fechada_por": reasons,
                "metricas_regra_lado": metrics,
                "recorte_lado": hand.get("_recorte_lado"),
                "recorte_px": hand.get("_recorte_px"),
            })
        return context

    def capture(track_id, keypoints, box=None, hand_context=None, observed_at=None):
        observed.append({
            "keypoints": [[float(value) for value in point] for point in keypoints],
            "box": [float(value) for value in box],
            "hand_context": {key: bool((hand_context or {}).get(key)) for key in HAND_KEYS},
            "maos": list(hands),
        })
        return analyze(track_id, keypoints, box, hand_context=hand_context, observed_at=observed_at)

    service._associate_hands = capture_hands
    service.analyzer.analyze = capture
    source = (_video_frames(cv2, args.video, args.inicio, args.segundos) if args.video
              else _webcam_frames(cv2, args.camera, args.segundos))
    frames = []
    for elapsed, image in source:
        observed.clear()
        frame_context = build_frame_context(image, PROCESS_SCALE, EXPERIMENTAL_GRAYSCALE)
        service.detect_gestures(frame_context)
        if observed:
            person = max(observed, key=lambda item: (item["box"][2] - item["box"][0])
                         * (item["box"][3] - item["box"][1]))
            frames.append({**person, "t": round(elapsed, 3)})
        else:
            frames.append({"keypoints": None, "t": round(elapsed, 3)})

    expected = sorted(GESTURES[args.gesto][2]) if args.gesto in GESTURES else []
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps({"gesto": args.gesto, "esperados": expected,
                                      "process_scale": PROCESS_SCALE,
                                      "maos_frame_cheio": args.maos_frame_cheio,
                                      "maos_recorte_punho": args.maos_recorte_punho,
                                      "fator_recorte_punho": (
                                          args.fator_recorte_punho
                                          if args.maos_recorte_punho else None
                                      ),
                                      "frames": frames}, indent=1), encoding="utf-8")
    duration = frames[-1]["t"] if frames else 0.0
    with_person = sum(1 for frame in frames if frame["keypoints"] is not None)
    print(f"{args.gesto}: {len(frames)} frames em {duration:.0f} s, pessoa em {with_person}")


if __name__ == "__main__":
    main()
