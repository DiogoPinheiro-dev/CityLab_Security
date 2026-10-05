import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "Server" / ".env", override=True)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "sim"}


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


MONGO_DETAILS = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "recon-db")
MONGO_SERVER_SELECTION_TIMEOUT_MS = _get_int("MONGO_SERVER_SELECTION_TIMEOUT_MS", 10000)
# Dias que um evento fica no banco; depois o proprio MongoDB apaga, por um
# indice TTL em data_hora_raw. Cada evento guarda um recorte em JPEG, e sem
# prazo o banco cresce sem parar. Zero guarda tudo. 30 por decisao do
# responsavel em 03/10/2026.
EVENT_RETENTION_DAYS = max(0, _get_int("EVENT_RETENTION_DAYS", 30))
# Endereco publico usado no QR code do cadastro; vazio deduz pelo request.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")

CITYLAB_ENABLE_FACE_SERVICE = _get_bool("CITYLAB_ENABLE_FACE_SERVICE", True)
CITYLAB_ENABLE_GESTURE_SERVICE = _get_bool("CITYLAB_ENABLE_GESTURE_SERVICE", True)
# Com True, a API sobe mesmo se um servico pesado falhar ao carregar.
CITYLAB_ALLOW_PARTIAL_PIPELINE = _get_bool("CITYLAB_ALLOW_PARTIAL_PIPELINE", False)

CITYLAB_PROFILE = os.getenv("CITYLAB_PROFILE", "default").strip().lower()
if CITYLAB_PROFILE not in {"default", "rpi3"}:
    raise ValueError("CITYLAB_PROFILE deve ser default ou rpi3")
_RPI3 = CITYLAB_PROFILE == "rpi3"

PROCESS_SCALE = max(0.1, min(1.0, _get_float("PROCESS_SCALE", 0.5)))
EXPERIMENTAL_GRAYSCALE = _get_bool("EXPERIMENTAL_GRAYSCALE", False)

FACE_MIN_WIDTH = _get_int("FACE_MIN_WIDTH", 40)
FACE_MIN_HEIGHT = _get_int("FACE_MIN_HEIGHT", 40)
FACE_MIN_CONFIDENCE = _get_float("FACE_MIN_CONFIDENCE", 0.45)
FACE_MINIMAL_MODULES = _get_bool("FACE_MINIMAL_MODULES", True)
FACE_PREFILTER = _get_bool("FACE_PREFILTER", True)
# Segundos em que um rosto no mesmo lugar herda o nome ja reconhecido, sem
# embedding; conhecido e desconhecido. Zero desliga. Exige FACE_PREFILTER.
# 15 no rpi3: o frame com uma pessoa ficou 13% mais rapido na media, medido
# em 01/10/2026 (docs/PLANO_OTIMIZACAO.md); no perfil default segue desligado.
FACE_REUSE_SECONDS = max(0.0, _get_float("FACE_REUSE_SECONDS", 15.0 if _RPI3 else 0.0))
# True gera o embedding no frame original, com os pontos do rosto achados na
# imagem reduzida pela PROCESS_SCALE: o ArcFace recorta os mesmos 112x112 de um
# rosto com o dobro de pixels. Exige FACE_PREFILTER. Ligado no rpi3: no Pi, em
# 03/10/2026, a semelhanca com o cadastro subiu, mais nos rostos dificeis, sem
# custo de tempo (docs/PLANO_OTIMIZACAO.md); no perfil default segue desligado.
FACE_EMBED_FULL_FRAME = _get_bool("FACE_EMBED_FULL_FRAME", _RPI3)
# Detector de rosto no lugar do det_10g do buffalo_l, em .onnx do InsightFace;
# o reconhecedor, e com ele o cadastro, ficam. Vazio usa o do pacote. O det_500m
# do buffalo_sc, no PC em 04/10/2026, detectou 6,7 vezes mais rapido, com os
# mesmos rostos no video de carga (docs/PLANO_OTIMIZACAO.md).
FACE_DETECTOR_PATH = os.getenv("FACE_DETECTOR_PATH", "").strip()
# Reconhecimento do rosto em segundo plano: a deteccao segue em todo frame, e o
# frame volta sem esperar o embedding, que roda numa thread propria; ate o nome
# sair, o rosto aparece como VERIFICANDO, sem evento. So com FACE_PREFILTER. Em
# avaliacao no Pi desde 04/10/2026, desligado nos dois perfis.
FACE_ASYNC_RECOGNITION = _get_bool("FACE_ASYNC_RECOGNITION", False)
# Aprende com o stream: um rosto reconhecido com folga vira mais uma referencia
# da pessoa, guardada no MongoDB (colecao rostos_aprendidos). Ate
# FACE_LEARNED_PER_PERSON por pessoa, uma a cada FACE_LEARN_INTERVAL_SECONDS, e
# o banco apaga as com mais de FACE_LEARNED_RETENTION_DAYS dias. Ligado no rpi3:
# no Pi, em 03/10/2026, com o video de carga, reconheceu 28 de 30 frames contra
# 26 nas tres rodadas, sem custo de tempo (docs/PLANO_OTIMIZACAO.md). Guarda
# rostos tirados do stream, dado biometrico na LGPD. Promovido aos dois perfis
# por decisao do responsavel em 05/10/2026; zero desliga.
FACE_LEARN_FROM_STREAM = _get_bool("FACE_LEARN_FROM_STREAM", True)
FACE_LEARNED_PER_PERSON = max(1, _get_int("FACE_LEARNED_PER_PERSON", 5))
FACE_LEARN_INTERVAL_SECONDS = max(0.0, _get_float("FACE_LEARN_INTERVAL_SECONDS", 600.0))
FACE_LEARNED_RETENTION_DAYS = max(0, _get_int("FACE_LEARNED_RETENTION_DAYS", 30))
# Zero preserva a escolha automatica da biblioteca. 2 no rpi3: com a pose em
# 416 o rosto virou o caminho critico, e 2 threads deixaram o frame com uma
# pessoa 20% mais rapido, medido em 01/10/2026 (docs/PLANO_OTIMIZACAO.md).
ONNX_INTRA_OP_THREADS = max(0, _get_int("ONNX_INTRA_OP_THREADS", 2 if _RPI3 else 0))
# False faz as threads intra-op do rosto dormirem entre operadores em vez de
# girar a espera (session.intra_op.allow_spinning=0). So vale com
# ONNX_INTRA_OP_THREADS acima de zero. Medido no Pi em 01/10/2026 sem efeito.
ONNX_ALLOW_SPINNING = _get_bool("ONNX_ALLOW_SPINNING", True)
# 3 no rpi3: com 2, a pose levava ~7,0 s; com 3, que o Ultralytics aplicava sem
# querer num dos workers, ~5,0 s (medido em 26/09/2026, docs/PLANO_OTIMIZACAO.md).
TORCH_NUM_THREADS = max(0, _get_int("TORCH_NUM_THREADS", 3 if _RPI3 else 0))
OPENCV_NUM_THREADS = max(0, _get_int("OPENCV_NUM_THREADS", 1 if _RPI3 else 0))
NATIVE_NUM_THREADS = max(0, _get_int("NATIVE_NUM_THREADS", 1 if _RPI3 else 0))
# Aceita arquivo .pt ou diretorio NCNN exportado; vazio usa o peso versionado.
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "").strip()
# Threads do ncnn na pose, so com um modelo NCNN. Zero deixa o padrao do ncnn,
# uma por nucleo fisico, 4 no Pi. A serie inicial com 4 teve subtensao causada
# pela tomada e nao compara threads. Com 2 no rpi3, a serie valida ficou 40%
# mais rapida que o .pt, com throttled=0x0 (docs/PLANO_OTIMIZACAO.md).
NCNN_NUM_THREADS = max(0, _get_int("NCNN_NUM_THREADS", 2 if _RPI3 else 0))
# Lado da entrada da pose, multiplo de 32. Zero mantem o padrao do Ultralytics,
# 640, que amplia o frame ja reduzido pela PROCESS_SCALE. 416 no rpi3: a pose
# caiu de 5,0 para 2,2 s e o frame com uma pessoa ficou 6% mais rapido, medido
# em 01/10/2026 (docs/PLANO_OTIMIZACAO.md); no perfil default segue 640.
POSE_IMGSZ = max(0, _get_int("POSE_IMGSZ", 416 if _RPI3 else 0))

PIPELINE_RUN_IN_PARALLEL = _get_bool("PIPELINE_RUN_IN_PARALLEL", True)
# Passada unica: pessoas saem da pose. Padrao no rpi3, medido nos tres cenarios
# em 26/09/2026; no perfil default segue opt-in porque nao foi medido.
PIPELINE_SHARED_PERSON_POSE = _get_bool("PIPELINE_SHARED_PERSON_POSE", _RPI3)
PIPELINE_MAX_WORKERS = max(1, _get_int("PIPELINE_MAX_WORKERS", 2))
GESTURE_ANALYZER_FPS = max(1, _get_int("GESTURE_ANALYZER_FPS", 12))
# Espera sem frames, excluindo o tempo gasto em inferencia.
GESTURE_IDLE_RESET_SECONDS = max(0.1, _get_float("GESTURE_IDLE_RESET_SECONDS", 5.0))
GESTURE_MAX_OBSERVATION_GAP_SECONDS = max(1.0, _get_float("GESTURE_MAX_OBSERVATION_GAP_SECONDS", 60.0))
# Caixa fraca alimenta o ByteTrack, mas nao deve sair publicada como pessoa.
# O track() do Ultralytics forca conf=0.1; a passada dupla usava 0,25.
GESTURE_PUBLISH_MIN_CONFIDENCE = max(0.0, min(1.0, _get_float("GESTURE_PUBLISH_MIN_CONFIDENCE", 0.25)))
# Gate de movimento antes da pose. Padrao no rpi3 junto com a passada unica.
GESTURE_MOTION_GATE = _get_bool("GESTURE_MOTION_GATE", _RPI3)
# Fracao de pixels alterados que ja conta como cena em movimento.
GESTURE_MOTION_MIN_RATIO = max(0.0, _get_float("GESTURE_MOTION_MIN_RATIO", 0.002))
GESTURE_MOTION_PIXEL_DELTA = max(1, _get_int("GESTURE_MOTION_PIXEL_DELTA", 25))
# Teto de tempo sem rodar a pose: protege quem entra em cena e fica parado.
GESTURE_MOTION_MAX_SKIP_SECONDS = max(0.0, _get_float("GESTURE_MOTION_MAX_SKIP_SECONDS", 30.0))

COOLDOWN_ALUNO_SECONDS = max(0.0, _get_float("COOLDOWN_ALUNO_SECONDS", 5.0))
COOLDOWN_NAO_ALUNO_SECONDS = max(0.0, _get_float("COOLDOWN_NAO_ALUNO_SECONDS", 5.0))
COOLDOWN_ALERTA_GESTO_SECONDS = max(0.0, _get_float("COOLDOWN_ALERTA_GESTO_SECONDS", 5.0))

STREAM_FPS = max(1, _get_int("STREAM_FPS", 10))
JPEG_QUALITY = max(0.1, min(1.0, _get_float("JPEG_QUALITY", 0.65)))
STREAM_WIDTH = max(160, _get_int("STREAM_WIDTH", 640))
STREAM_HEIGHT = max(120, _get_int("STREAM_HEIGHT", 480))
MAX_IN_FLIGHT_FRAMES = max(1, _get_int("MAX_IN_FLIGHT_FRAMES", 1 if _RPI3 else 2))

DEBUG_PIPELINE = _get_bool("DEBUG_PIPELINE", False)
ENABLE_PERFORMANCE_METRICS = _get_bool("ENABLE_PERFORMANCE_METRICS", False)
ENABLE_SYSTEM_MONITOR = _get_bool("ENABLE_SYSTEM_MONITOR", False)
SYSTEM_MONITOR_INTERVAL_SECONDS = max(
    1.0,
    _get_float("SYSTEM_MONITOR_INTERVAL_SECONDS", 5.0),
)
