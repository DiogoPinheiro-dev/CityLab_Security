"""Executor offline do TCC de PDI: roda um video pela cadeia unificada, sem servidor.

Le a galeria do MongoDB uma vez, so leitura, e nao grava nada no banco. Pega um
quadro a cada --intervalo segundos de video (1,7 s, o ritmo do Pi), aplica a
configuracao de PDI antes da reducao e passa a cadeia o instante do quadro no
video. Grava um JSONL: cabecalho da rodada, um registro por quadro e um resumo.
Uma rodada por processo: o rastreador e o historico comecam do zero.

Exemplo:
    .venv\\Scripts\\python.exe tools\\executor_pdi.py video.avi --pdi clahe \\
        --saida resultados\\tcc-pdi\\clahe-r1.jsonl
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configuracao do experimento, decidida em docs/PLANO_TCC_PDI.md. Vale sobre o
# .env: perfil do Pi, os dois servicos, sem aprendizado nem segundo plano, que
# fariam a saida depender de rodadas anteriores ou do escalonamento.
EXPERIMENT_ENV = {
    "CITYLAB_PROFILE": "rpi3",
    "CITYLAB_ENABLE_FACE_SERVICE": "1",
    "CITYLAB_ENABLE_GESTURE_SERVICE": "1",
    "CITYLAB_ALLOW_PARTIAL_PIPELINE": "0",
    "FACE_LEARN_FROM_STREAM": "0",
    "FACE_ASYNC_RECOGNITION": "0",
    "ENABLE_PERFORMANCE_METRICS": "1",
    # Sem rede: pacote faltando deve parar a rodada, nao ser instalado no meio.
    "YOLO_AUTOINSTALL": "False",
}
DEFAULT_POSE_MODEL = ROOT / "App" / "GestureRecon" / "yolov8n-pose_ncnn_model"
DEFAULT_FACE_DETECTOR = Path.home() / ".insightface" / "models" / "buffalo_sc" / "det_500m.onnx"
RECOGNIZER = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
HAND_MODEL = ROOT / "App" / "GestureRecon" / "models" / "gesture_recognizer.task"
SETTINGS_RECORDED = (
    "CITYLAB_PROFILE", "PROCESS_SCALE", "PIPELINE_RUN_IN_PARALLEL", "PIPELINE_MAX_WORKERS",
    "PIPELINE_SHARED_PERSON_POSE", "GESTURE_MOTION_GATE", "POSE_MODEL_PATH", "POSE_IMGSZ",
    "NCNN_NUM_THREADS", "ONNX_INTRA_OP_THREADS", "TORCH_NUM_THREADS", "OPENCV_NUM_THREADS",
    "NATIVE_NUM_THREADS", "FACE_DETECTOR_PATH", "FACE_REUSE_SECONDS", "FACE_EMBED_FULL_FRAME",
    "FACE_ASYNC_RECOGNITION", "FACE_LEARN_FROM_STREAM", "FACE_PREFILTER", "FACE_MIN_WIDTH",
    "FACE_MIN_CONFIDENCE", "GESTURE_PUBLISH_MIN_CONFIDENCE", "MONGO_DB_NAME",
)
PACKAGES = ("numpy", "opencv-contrib-python", "opencv-python", "mediapipe", "insightface",
            "onnxruntime", "ultralytics", "torch", "ncnn", "lapx", "psutil", "pymongo")


def apply_experiment_env(pose_model, face_detector, environ=os.environ):
    """Antes de importar App.settings: o ambiente ja definido vence o .env."""
    environ.update(EXPERIMENT_ENV)
    environ["POSE_MODEL_PATH"] = str(pose_model)
    environ["FACE_DETECTOR_PATH"] = str(face_detector)


def sample_frames(fps, frame_count, interval, start):
    """Indices e instantes, no video, de um quadro a cada `interval` segundos."""
    if fps <= 0 or interval <= 0 or start < 0:
        raise ValueError("fps e intervalo precisam ser positivos, e o inicio nao negativo")
    samples, instant = [], start
    while True:
        index = round(instant * fps)
        if index >= frame_count:
            return samples
        samples.append((index, index / fps))
        instant += interval


def gallery_from_documents(documents):
    """Nomes, vetores e um hash da galeria, que identifica a usada na rodada."""
    import numpy as np

    names, vectors = [], []
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: (item["nome"], item["embedding"])):
        vector = np.asarray(document["embedding"], dtype=np.float32)
        names.append(document["nome"])
        vectors.append(vector)
        digest.update(document["nome"].encode("utf-8"))
        digest.update(vector.tobytes())
    return names, vectors, digest.hexdigest()


def face_records(faces):
    return [{"nome": face.get("name"), "bbox": face.get("bbox"),
             "semelhanca": round(float(face.get("confidence", 0.0)), 4),
             "deteccao": round(float(face.get("det_confidence", 0.0)), 4),
             "pendente": bool(face.get("pending", False))} for face in faces]


def gesture_records(gestures):
    return [{"track_id": gesture.get("track_id"), "bbox": gesture.get("bbox"),
             "alertas": list(gesture.get("alerts") or [])} for gesture in gestures]


def summarize(values):
    if not values:
        return None
    return {"n": len(values), "media": statistics.fmean(values),
            "desvio": statistics.stdev(values) if len(values) > 1 else 0.0,
            "mediana": statistics.median(values), "minimo": min(values), "maximo": max(values)}


def file_sha256(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def models_sha256(pose_model, face_detector):
    files = {"detector_rosto": Path(face_detector), "reconhecedor_rosto": RECOGNIZER,
             "maos": HAND_MODEL}
    pose = Path(pose_model)
    for item in sorted(pose.iterdir()) if pose.is_dir() else [pose]:
        if item.suffix in {".param", ".bin", ".pt"}:
            files[f"pose/{item.name}"] = item
    return {name: file_sha256(path) if path.is_file() else None for name, path in files.items()}


def git_state():
    def git(*args):
        return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
                              cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "alteracoes_sem_commit": bool(git("status", "--porcelain", "--untracked-files=no"))}


def machine(psutil):
    return {"sistema": platform.platform(), "processador": platform.processor(),
            "nucleos_fisicos": psutil.cpu_count(logical=False),
            "nucleos_logicos": psutil.cpu_count(logical=True),
            "memoria_gb": round(psutil.virtual_memory().total / 2**30, 2),
            "python": platform.python_version()}


def versions():
    found = {}
    for package in PACKAGES:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            found[package] = None
    return found


def load_gallery(settings):
    from pymongo import MongoClient

    client = MongoClient(settings.MONGO_DETAILS,
                         serverSelectionTimeoutMS=settings.MONGO_SERVER_SELECTION_TIMEOUT_MS)
    try:
        documents = list(client[settings.MONGO_DB_NAME]["alunos"].find(
            {}, {"_id": 0, "nome": 1, "embedding": 1}))
    finally:
        client.close()
    if not documents:
        raise RuntimeError("Nenhum cadastro na colecao alunos: a galeria ficaria vazia")
    return gallery_from_documents(documents)


def resize_to_width(frame, width, cv2):
    height, current = frame.shape[:2]
    if width <= 0 or current == width:
        return frame
    return cv2.resize(frame, (width, round(height * width / current)), interpolation=cv2.INTER_AREA)


def reset_run_state(pipeline):
    """Depois do aquecimento: historico de gesto, nomes e rastreador do zero."""
    pipeline.reset_gesture_history()
    pipeline.reset_face_identities()
    predictor = getattr(getattr(pipeline.gesture_service, "pose_model", None), "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        tracker.reset()


def run(args):
    apply_experiment_env(args.pose_model, args.face_detector)
    from App import settings
    from App.inference_runtime import prepare_native_environment

    prepare_native_environment(settings.NATIVE_NUM_THREADS)
    import cv2
    import psutil
    # O InsightFace (skimage) e o rastreador da pose importam o scipy no primeiro
    # quadro, cada um na sua thread; ao mesmo tempo, o import circular quebra a
    # rodada. Importar antes, na thread principal.
    import scipy.spatial  # noqa: F401
    import skimage.transform  # noqa: F401

    from App.preprocessing import Preprocessor
    from App.recognition_pipeline import UnifiedRecognitionService

    preprocessor = Preprocessor.from_spec(args.pdi)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = sample_frames(fps, frame_count, args.intervalo, args.inicio)
    if not samples:
        raise RuntimeError("O video e mais curto que o inicio pedido")

    names, embeddings, gallery_hash = load_gallery(settings)
    pipeline = UnifiedRecognitionService(preprocessor=preprocessor)
    pipeline.face_service.replace_known_faces(names, embeddings)
    process = psutil.Process()

    header = {
        "tipo": "cabecalho", "rodada": args.rodada or uuid.uuid4().hex,
        "inicio_utc": datetime.now(timezone.utc).isoformat(), **git_state(),
        "pdi": args.pdi, "pdi_config": preprocessor.config,
        "video": {"arquivo": Path(args.video).name, "sha256": file_sha256(args.video), "fps": fps,
                  "quadros": frame_count,
                  "resolucao": [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]},
        "amostragem": {"intervalo_s": args.intervalo, "inicio_s": args.inicio,
                       "quadros": len(samples), "largura_entrada": args.largura,
                       "aquecimento": args.aquecimento},
        "galeria": {"cadastros": len(names), "sha256": gallery_hash},
        "configuracao": {name: getattr(settings, name) for name in SETTINGS_RECORDED},
        "modelos_sha256": models_sha256(args.pose_model, args.face_detector),
        "versoes": versions(), "maquina": machine(psutil),
    }
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    totals, pdi_times = [], []
    with args.saida.open("w", encoding="utf-8") as output:
        output.write(json.dumps(header, ensure_ascii=False) + "\n")
        position = 0
        for number, (index, instant) in enumerate(samples):
            read_started = time.perf_counter()
            while position <= index:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Video terminou antes do quadro {index}")
                position += 1
            frame = resize_to_width(frame, args.largura, cv2)
            read_ms = (time.perf_counter() - read_started) * 1000.0
            if number == 0:
                # Carga e compilacao dos modelos fora da medicao.
                for warmup in range(args.aquecimento):
                    pipeline.process_frame(frame.copy(), observed_at=-1000.0 + warmup)
                reset_run_state(pipeline)
                process.cpu_percent(None)
                cpu_before, io_before = process.cpu_times(), process.io_counters()
            started = time.perf_counter()
            payload = pipeline.process_frame(frame, observed_at=instant)
            frame_ms = (time.perf_counter() - started) * 1000.0
            cpu_after, io_after = process.cpu_times(), process.io_counters()
            metrics = payload.get("metrics", {})
            record = {
                "tipo": "quadro", "amostra": number, "quadro_video": index, "t": round(instant, 4),
                "leitura_ms": round(read_ms, 3), "quadro_ms": round(frame_ms, 3),
                "rostos": face_records(payload.get("faces", [])),
                "pessoas": len(payload.get("persons", [])),
                "gestos": gesture_records(payload.get("gestures", [])),
                "metricas": metrics,
                "processo": {
                    "cpu_percent": process.cpu_percent(None),
                    "cpu_ms": round(((cpu_after.user + cpu_after.system)
                                     - (cpu_before.user + cpu_before.system)) * 1000.0, 3),
                    "rss_mb": round(process.memory_info().rss / 2**20, 1),
                    "leitura_bytes": io_after.read_bytes - io_before.read_bytes,
                    "escrita_bytes": io_after.write_bytes - io_before.write_bytes,
                },
            }
            cpu_before, io_before = cpu_after, io_after
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            totals.append(metrics.get("total_ms", frame_ms))
            if "pdi_ms" in metrics:
                pdi_times.append(metrics["pdi_ms"])
        output.write(json.dumps({"tipo": "resumo", "quadros": len(totals),
                                 "total_ms": summarize(totals), "pdi_ms": summarize(pdi_times),
                                 "fim_utc": datetime.now(timezone.utc).isoformat()},
                                ensure_ascii=False) + "\n")
    capture.release()
    pipeline.close()
    print(f"{len(totals)} quadros; resultado: {args.saida}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--pdi", default="nenhuma",
                        help="Configuracao de PDI, ex.: nenhuma, clahe, gama:gamma=0.6+gaussiano")
    parser.add_argument("--saida", type=Path, required=True, help="Arquivo JSONL da rodada")
    parser.add_argument("--intervalo", type=float, default=1.7,
                        help="Segundos de video entre quadros (1,7 = ritmo do Pi)")
    parser.add_argument("--inicio", type=float, default=0.0,
                        help="Instante do primeiro quadro, em segundos (fase da amostragem)")
    parser.add_argument("--largura", type=int, default=640,
                        help="Largura do quadro que entra na cadeia, mantendo a proporcao; 0 = original")
    parser.add_argument("--aquecimento", type=int, default=3,
                        help="Passadas no primeiro quadro antes da medicao")
    parser.add_argument("--rodada", help="Identificador da rodada; sem ele, um aleatorio")
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--face-detector", type=Path, default=DEFAULT_FACE_DETECTOR)
    args = parser.parse_args(argv)
    if not args.video.is_file():
        parser.error(f"Video nao encontrado: {args.video}")
    for option, path in (("--pose-model", args.pose_model), ("--face-detector", args.face_detector)):
        if not path.exists():
            parser.error(f"{option} nao encontrado: {path}")
    run(args)


if __name__ == "__main__":
    main()
