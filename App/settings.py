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
# Zero preserva a escolha automatica da biblioteca.
ONNX_INTRA_OP_THREADS = max(0, _get_int("ONNX_INTRA_OP_THREADS", 1 if _RPI3 else 0))
TORCH_NUM_THREADS = max(0, _get_int("TORCH_NUM_THREADS", 2 if _RPI3 else 0))
OPENCV_NUM_THREADS = max(0, _get_int("OPENCV_NUM_THREADS", 1 if _RPI3 else 0))
NATIVE_NUM_THREADS = max(0, _get_int("NATIVE_NUM_THREADS", 1 if _RPI3 else 0))
# Aceita arquivo .pt ou diretorio NCNN exportado; vazio usa o peso versionado.
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "").strip()

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
