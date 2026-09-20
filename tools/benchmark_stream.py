"""Mede o stream com video fixo ou webcam, um frame pendente por vez."""
import argparse
import asyncio
import hashlib
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


def summarize(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "median": None, "p95": None}
    return {"count": len(values), "median": statistics.median(values),
            "p95": values[max(0, math.ceil(len(values) * .95) - 1)]}


def capture_property(capture, property_id):
    value = capture.get(property_id)
    return value if math.isfinite(value) and value > 0 else None


def detection_confidences(payload):
    # Confiancas de caixas, sem nomes, imagens ou embeddings.
    return {output: [item["confidence"] for item in payload.get(source, [])
                     if isinstance(item.get("confidence"), (int, float))
                     and not isinstance(item["confidence"], bool)
                     and math.isfinite(item["confidence"])]
            for output, source in (("persons_confidence", "pessoas"),
                                   ("gestures_confidence", "gestos"))}


def open_source(args, cv2):
    if args.camera_index is not None:
        capture = cv2.VideoCapture(args.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir a webcam {args.camera_index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        source = {
            "source_type": "webcam", "camera_index": args.camera_index,
            "capture_width": capture_property(capture, cv2.CAP_PROP_FRAME_WIDTH),
            "capture_height": capture_property(capture, cv2.CAP_PROP_FRAME_HEIGHT),
            "capture_fps": capture_property(capture, cv2.CAP_PROP_FPS),
            "video_sha256": None,
        }
    else:
        with args.video.open("rb") as video:
            digest = hashlib.file_digest(video, "sha256").hexdigest()
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError("Nao foi possivel abrir o video")
        source = {"source_type": "video", "video_sha256": digest}
    return capture, source


async def run(args):
    import cv2
    from websockets.asyncio.client import connect

    capture, source = open_source(args, cv2)
    run_started_at_utc = datetime.now(timezone.utc).isoformat()
    rows = []
    sent_bytes = received_bytes = 0
    try:
        async with connect(args.url, max_size=4 * 1024 * 1024, compression=None) as ws:
            started = None
            for index in range(args.warmup + args.frames):
                ok, frame = capture.read()
                if not ok:
                    if args.camera_index is not None:
                        raise RuntimeError("Falha ao ler frame da webcam")
                    raise RuntimeError("Video insuficiente para warmup + frames; use a mesma carga em todas as execucoes")
                frame = cv2.resize(frame, (args.width, args.height))
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if not ok:
                    raise RuntimeError("Falha ao codificar JPEG")
                if index == args.warmup:
                    started = time.perf_counter()
                sent_at = time.perf_counter()
                await ws.send(jpeg.tobytes())
                raw = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
                received_at = time.perf_counter()
                payload = json.loads(raw)
                if "erro" in payload:
                    raise RuntimeError(payload["erro"])
                if not payload.get("metrics"):
                    raise RuntimeError("Ative ENABLE_PERFORMANCE_METRICS=1 no servidor")
                if index >= args.warmup:
                    gestures = payload.get("gestos") or []
                    rows.append({"frame": index, "rtt_ms": (received_at - sent_at) * 1000,
                                 "persons_count": len(payload.get("pessoas") or []),
                                 "faces_count": len(payload.get("rostos") or []),
                                 "gestures_count": len(gestures),
                                 "alerts_count": sum(len(item.get("alerts") or []) for item in gestures),
                                 **detection_confidences(payload),
                                 "metrics": payload["metrics"]})
                    sent_bytes += len(jpeg)
                    received_bytes += len(raw.encode("utf-8") if isinstance(raw, str) else raw)
            elapsed = time.perf_counter() - started
    finally:
        capture.release()
    keys = sorted({key for row in rows for key in row["metrics"]})
    report = {
        "scenario": args.scenario, **source,
        "started_at_utc": run_started_at_utc,
        "client_python": platform.python_version(), "client_platform": platform.platform(),
        "server_run_label": args.run_label,
        "config": {"width": args.width, "height": args.height, "jpeg_quality": args.quality,
                   "warmup_frames": args.warmup, "frames": args.frames, "max_in_flight": 1},
        "elapsed_seconds": elapsed, "completed_fps": len(rows) / elapsed,
        "rtt_ms": summarize([row["rtt_ms"] for row in rows]),
        "detections": {key: summarize([row[key] for row in rows]) for key in
                       ("persons_count", "faces_count", "gestures_count", "alerts_count")},
        "metrics": {key: summarize([row["metrics"][key] for row in rows
                                    if isinstance(row["metrics"].get(key), (float, int))]) for key in keys},
        "application_bytes_sent": sent_bytes, "application_bytes_received": received_bytes,
        "application_bytes_per_second": (sent_bytes + received_bytes) / elapsed,
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} frames; {report['completed_fps']:.2f} FPS; resultado: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--camera-index", type=int, help="Indice da webcam local, por exemplo 0")
    parser.add_argument("--scenario", required=True, choices=["empty", "one-person", "many-persons"])
    parser.add_argument("--run-label", required=True, help="Identificador do ambiente/configuracao do servidor")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/stream")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--quality", type=int, default=65)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if (args.video is None) == (args.camera_index is None):
        parser.error("Informe um video ou --camera-index")
    if args.camera_index is not None and args.camera_index < 0:
        parser.error("--camera-index deve ser zero ou maior")
    if min(args.frames, args.width, args.height, args.timeout) <= 0 or args.warmup < 0 or not 1 <= args.quality <= 100:
        parser.error("Dimensoes, frames e timeout devem ser positivos; warmup >= 0; quality entre 1 e 100")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
