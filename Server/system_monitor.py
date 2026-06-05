import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from App.settings import SYSTEM_MONITOR_INTERVAL_SECONDS

try:
    import psutil
except ImportError:
    psutil = None


class SystemMonitor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.interval_seconds = SYSTEM_MONITOR_INTERVAL_SECONDS
        self.last_snapshot_at = 0.0
        self.frame_times_ms: deque[float] = deque(maxlen=120)
        self.fps_values: deque[float] = deque(maxlen=120)

        self.logger = logging.getLogger("SystemMonitor")
        self.logger.setLevel(logging.INFO)
        if enabled and not self.logger.handlers:
            log_dir = Path(__file__).resolve().parent / "historico"
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_dir / "monitoramento_raspberry.log", mode="a", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(handler)

    def record_frame_metrics(self, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return

        total_ms = self._to_float(metrics.get("total_ms"))
        fps = self._to_float(metrics.get("effective_fps"))
        if total_ms is not None:
            self.frame_times_ms.append(total_ms)
        if fps is not None:
            self.fps_values.append(fps)

    def maybe_log_snapshot(self) -> None:
        if not self.enabled:
            return

        now = time.time()
        if (now - self.last_snapshot_at) < self.interval_seconds:
            return

        self.last_snapshot_at = now
        cpu_percent = psutil.cpu_percent(interval=None) if psutil else None
        ram_percent = psutil.virtual_memory().percent if psutil else None
        temperature = self._read_temperature()
        avg_frame_ms = self._average(self.frame_times_ms)
        avg_fps = self._average(self.fps_values)

        self.logger.info(
            "cpu=%s ram=%s temp_c=%s avg_fps=%s avg_frame_ms=%s",
            self._fmt(cpu_percent),
            self._fmt(ram_percent),
            self._fmt(temperature),
            self._fmt(avg_fps),
            self._fmt(avg_frame_ms),
        )

    def _read_temperature(self) -> float | None:
        thermal_zone = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal_zone.exists():
            try:
                return int(thermal_zone.read_text(encoding="utf-8").strip()) / 1000.0
            except (OSError, ValueError):
                return None

        if psutil and hasattr(psutil, "sensors_temperatures"):
            try:
                sensors = psutil.sensors_temperatures()
            except Exception:
                return None
            for values in sensors.values():
                if values:
                    return float(values[0].current)
        return None

    def _average(self, values: deque[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    def _fmt(self, value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.2f}"

    def _to_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
