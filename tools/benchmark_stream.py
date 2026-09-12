"""Replay a fixed video against the real API, with bounded memory and one pending frame."""
import argparse
import asyncio
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path


def summarize(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "median": None, "p95": None}
    return {"count": len(values), "median": statistics.median(values),
            "p95": values[max(0, math.ceil(len(values) * .95) - 1)]}


async def run(args):
    import cv2
    from websockets.asyncio.client import connect

    with args.video.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError("Nao foi possivel abrir o video")
    rows = []
    sent_bytes = received_bytes = 0
    try:
        async with connect(args.url, max_size=4 * 1024 * 1024, compression=None) as ws:
            started = None
            for index in range(args.warmup + args.frames):
                ok, frame = capture.read()
                if not ok:
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
                    rows.append({"frame": index, "rtt_ms": (received_at - sent_at) * 1000,
                                 "metrics": payload["metrics"]})
                    sent_bytes += len(jpeg)
                    received_bytes += len(raw.encode("utf-8") if isinstance(raw, str) else raw)
            elapsed = time.perf_counter() - started
    finally:
        capture.release()
    keys = sorted({key for row in rows for key in row["metrics"]})
    report = {
        "scenario": args.scenario, "video_sha256": digest,
        "client_python": platform.python_version(), "client_platform": platform.platform(),
        "server_run_label": args.run_label,
        "config": {"width": args.width, "height": args.height, "jpeg_quality": args.quality,
                   "warmup_frames": args.warmup, "frames": args.frames, "max_in_flight": 1},
        "elapsed_seconds": elapsed, "completed_fps": len(rows) / elapsed,
        "rtt_ms": summarize([row["rtt_ms"] for row in rows]),
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
    parser.add_argument("video", type=Path)
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
    if min(args.frames, args.width, args.height, args.timeout) <= 0 or args.warmup < 0 or not 1 <= args.quality <= 100:
        parser.error("Dimensoes, frames e timeout devem ser positivos; warmup >= 0; quality entre 1 e 100")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
