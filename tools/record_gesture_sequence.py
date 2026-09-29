"""Grava um gesto real para o banco de replay, sem imagem.

Roda a pose e as maos do projeto na webcam local, ou num video gravado, e
guarda por frame so o que o GestureAnalyzer recebe: keypoints, caixa e o
estado das maos. Nao grava imagem, rosto nem nome. Uma pessoa por vez na
frente da camera; com mais de uma, fica a maior caixa. Feche antes a pagina
do stream, que usa a webcam.

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
    args = parser.parse_args()

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
    service.pose_model.track = (lambda frame, persist=True, tracker=None, **options:
                                predict(frame, **options))
    observed = []
    analyze = service.analyzer.analyze

    def capture(track_id, keypoints, box=None, hand_context=None, observed_at=None):
        observed.append({
            "keypoints": [[float(value) for value in point] for point in keypoints],
            "box": [float(value) for value in box],
            "hand_context": {key: bool((hand_context or {}).get(key)) for key in HAND_KEYS},
        })
        return analyze(track_id, keypoints, box, hand_context=hand_context, observed_at=observed_at)

    service.analyzer.analyze = capture
    source = (_video_frames(cv2, args.video, args.inicio, args.segundos) if args.video
              else _webcam_frames(cv2, args.camera, args.segundos))
    frames = []
    for elapsed, image in source:
        observed.clear()
        service.detect_gestures(build_frame_context(image, PROCESS_SCALE, EXPERIMENTAL_GRAYSCALE))
        if observed:
            person = max(observed, key=lambda item: (item["box"][2] - item["box"][0])
                         * (item["box"][3] - item["box"][1]))
            frames.append({**person, "t": round(elapsed, 3)})
        else:
            frames.append({"keypoints": None, "t": round(elapsed, 3)})

    expected = sorted(GESTURES[args.gesto][2]) if args.gesto in GESTURES else []
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps({"gesto": args.gesto, "esperados": expected,
                                      "frames": frames}, indent=1), encoding="utf-8")
    duration = frames[-1]["t"] if frames else 0.0
    with_person = sum(1 for frame in frames if frame["keypoints"] is not None)
    print(f"{args.gesto}: {len(frames)} frames em {duration:.0f} s, pessoa em {with_person}")


if __name__ == "__main__":
    main()
