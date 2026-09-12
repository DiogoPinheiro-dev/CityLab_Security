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


PROCESS_SCALE = max(0.1, min(1.0, _get_float("PROCESS_SCALE", 0.5)))
EXPERIMENTAL_GRAYSCALE = _get_bool("EXPERIMENTAL_GRAYSCALE", False)

FACE_MIN_WIDTH = _get_int("FACE_MIN_WIDTH", 40)
FACE_MIN_HEIGHT = _get_int("FACE_MIN_HEIGHT", 40)
FACE_MIN_CONFIDENCE = _get_float("FACE_MIN_CONFIDENCE", 0.45)

PIPELINE_RUN_IN_PARALLEL = _get_bool("PIPELINE_RUN_IN_PARALLEL", True)
PIPELINE_MAX_WORKERS = max(1, _get_int("PIPELINE_MAX_WORKERS", 2))
GESTURE_ANALYZER_FPS = max(1, _get_int("GESTURE_ANALYZER_FPS", 12))

COOLDOWN_ALUNO_SECONDS = max(0.0, _get_float("COOLDOWN_ALUNO_SECONDS", 5.0))
COOLDOWN_NAO_ALUNO_SECONDS = max(0.0, _get_float("COOLDOWN_NAO_ALUNO_SECONDS", 5.0))
COOLDOWN_ALERTA_GESTO_SECONDS = max(0.0, _get_float("COOLDOWN_ALERTA_GESTO_SECONDS", 5.0))

STREAM_FPS = max(1, _get_int("STREAM_FPS", 10))
JPEG_QUALITY = max(0.1, min(1.0, _get_float("JPEG_QUALITY", 0.65)))
STREAM_WIDTH = max(160, _get_int("STREAM_WIDTH", 640))
STREAM_HEIGHT = max(120, _get_int("STREAM_HEIGHT", 480))
MAX_IN_FLIGHT_FRAMES = max(1, _get_int("MAX_IN_FLIGHT_FRAMES", 2))

DEBUG_PIPELINE = _get_bool("DEBUG_PIPELINE", False)
ENABLE_PERFORMANCE_METRICS = _get_bool("ENABLE_PERFORMANCE_METRICS", False)
ENABLE_SYSTEM_MONITOR = _get_bool("ENABLE_SYSTEM_MONITOR", False)
SYSTEM_MONITOR_INTERVAL_SECONDS = max(
    1.0,
    _get_float("SYSTEM_MONITOR_INTERVAL_SECONDS", 5.0),
)
